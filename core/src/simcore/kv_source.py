"""KV 传输来源：按提交顺序串行读取，返回各组数据到齐时刻。

同一来源的读取先来先服务（与 SGLang HiCache 单个 prefetch IO 线程串行处理各操作一致），
每组读取计一次固定延迟。可选的 SSD 式波动由“正常 ↔ 降速”两状态交替与每批读取带宽小幅
抖动组成：正常段时长服从指数分布，降速段时长服从对数正态分布（变异系数控制长尾）。
按固定种子生成以保证同一配置结果可复现。
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from simcore.schemas import L3Fluctuation

__all__ = [
    "IDLE_RESTART_CYCLES",
    "STORAGE_BATCH_PAGES",
    "KvSource",
    "KvSourceSpec",
]

# SGLang HiCache 每次存储读取的页数（mem_cache/hicache_storage.py: STORAGE_BATCH_SIZE = 128）。
STORAGE_BATCH_PAGES = 128

# 链路空档超过该数量的平均“正常 + 降速”周期时，直接按稳态重新抽取状态，不逐段推进。
IDLE_RESTART_CYCLES = 1000


def _lognormal(mean: float, cv: float) -> tuple[float, float]:  # 均值与变异系数 → (mu, sigma)
    variance = math.log1p(cv * cv)
    return math.log(mean) - variance / 2, math.sqrt(variance)


@dataclass(frozen=True, slots=True)
class KvSourceSpec:
    """来源的不可变规格；``fluctuation`` 为 None 时按固定延迟与带宽确定性计时。"""

    bytes_per_ms: float
    latency_ms: float
    fluctuation: L3Fluctuation | None = None


class KvSource:
    """单个来源的链路占用与波动状态；读取开始时刻单调不减，状态序列随时间按需生成。"""

    __slots__ = (
        "_bytes_per_ms",
        "_degraded",
        "_degraded_factor",
        "_degraded_latency_ms",
        "_degraded_mean_ms",
        "_degraded_mu",
        "_degraded_sigma",
        "_fluctuating",
        "_fraction",
        "_jitter_mu",
        "_jitter_sigma",
        "_latency_ms",
        "_normal_rate",
        "_restart_gap_ms",
        "_rng",
        "_state_end",
        "free_ms",
    )

    def __init__(self, spec: KvSourceSpec, seed: str) -> None:  # 来源规格与种子 → 空闲来源
        self._bytes_per_ms = spec.bytes_per_ms
        self._latency_ms = spec.latency_ms
        self.free_ms = 0.0  # 链路空闲时刻
        self._rng = random.Random(seed)
        fluctuation = spec.fluctuation
        self._fluctuating = fluctuation is not None
        self._degraded = False
        self._state_end = math.inf  # 不降速时状态永不切换
        self._fraction = self._normal_rate = self._degraded_mean_ms = 0.0
        self._degraded_mu = self._degraded_sigma = 0.0
        self._degraded_factor = 1.0
        self._degraded_latency_ms = self._jitter_mu = self._jitter_sigma = 0.0
        self._restart_gap_ms = math.inf
        if fluctuation is None:
            return
        self._degraded_factor = fluctuation.degraded_bandwidth_factor
        self._degraded_latency_ms = fluctuation.degraded_extra_latency_ms
        self._jitter_mu, self._jitter_sigma = _lognormal(1.0, fluctuation.jitter_cv)
        fraction = fluctuation.degraded_fraction
        if fraction > 0:
            mean = fluctuation.mean_degraded_ms
            self._fraction = fraction
            self._degraded_mean_ms = mean
            self._normal_rate = fraction / (mean * (1 - fraction))  # 正常段平均时长的倒数
            self._degraded_mu, self._degraded_sigma = _lognormal(
                mean, fluctuation.degraded_duration_cv
            )
            self._restart_gap_ms = IDLE_RESTART_CYCLES * mean / fraction
            self._restart(0.0)  # 自稳态开始

    def submit(
        self, start_ms: float, sizes: Sequence[float]
    ) -> list[float]:  # 提交时刻与各组字节数 → 各组按序读完的绝对时刻
        """各组依次读取，均不早于前一读取完成；返回后链路占用至最后一组完成。"""
        now = start_ms if start_ms > self.free_ms else self.free_ms
        ends: list[float] = []
        if self._fluctuating:
            for size in sizes:
                now = self._read(now, size)
                ends.append(now)
        else:
            latency, rate = self._latency_ms, self._bytes_per_ms
            for size in sizes:
                now += latency + size / rate
                ends.append(now)
        self.free_ms = now
        return ends

    def _segment_ms(self) -> float:  # 刚进入的状态 → 该段时长（ms）
        if not self._degraded:
            return self._rng.expovariate(self._normal_rate)
        if self._degraded_sigma:
            return self._rng.lognormvariate(self._degraded_mu, self._degraded_sigma)
        return self._degraded_mean_ms

    def _restart(self, now: float) -> None:
        """按稳态抽取 ``now`` 时刻的状态及其剩余时长。

        处于降速的概率为时间占比；所处降速段按长度加权抽取（对数正态的长度加权分布为
        mu 增加 sigma² 的对数正态），剩余时长为其均匀比例。正常段指数分布无记忆。
        """
        rng = self._rng
        self._degraded = rng.random() < self._fraction
        if self._degraded:
            sigma = self._degraded_sigma
            length = (
                rng.lognormvariate(self._degraded_mu + sigma * sigma, sigma)
                if sigma
                else self._degraded_mean_ms
            )
            residual = rng.random() * length
        else:
            residual = rng.expovariate(self._normal_rate)
        self._state_end = now + residual

    def _advance(self, now: float, *, idle: bool = False) -> None:  # 推进状态至覆盖 now
        if now < self._state_end:
            return
        if idle and now - self._state_end > self._restart_gap_ms:
            self._restart(now)  # 长空档后状态已与此前无关，直接取稳态
            return
        while now >= self._state_end:
            self._degraded = not self._degraded
            self._state_end += self._segment_ms()

    def _read(self, now: float, size: float) -> float:  # 开始时刻与字节数 → 波动下的完成时刻
        self._advance(now, idle=True)
        now += self._latency_ms + (self._degraded_latency_ms if self._degraded else 0.0)
        self._advance(now, idle=True)
        rate = self._bytes_per_ms
        if self._jitter_sigma:
            rate *= self._rng.lognormvariate(self._jitter_mu, self._jitter_sigma)
        remaining = size
        while True:  # 跨越状态切换时分段按各段带宽传输
            self._advance(now)
            current = rate * self._degraded_factor if self._degraded else rate
            capacity = current * (self._state_end - now)
            if remaining <= capacity:
                return now + remaining / current
            remaining -= capacity
            now = self._state_end
