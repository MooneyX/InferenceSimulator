"""KV 传输来源：按提交顺序串行读取，返回各组数据到齐时刻。

同一来源的读取先来先服务（与 SGLang HiCache 单个 prefetch IO 线程串行处理各操作一致），
每组读取计一次固定延迟。可选的波动模型由“正常 ↔ 降速”两状态马尔可夫切换与单次读取
带宽小幅抖动组成，按固定种子生成以保证同一配置结果可复现。
"""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "L3_FLUCTUATION",
    "STORAGE_BATCH_PAGES",
    "Fluctuation",
    "KvSource",
    "KvSourceSpec",
]

# SGLang HiCache 每次存储读取的页数（mem_cache/hicache_storage.py: STORAGE_BATCH_SIZE = 128）。
STORAGE_BATCH_PAGES = 128


@dataclass(frozen=True, slots=True)
class Fluctuation:
    """两状态降速与单次抖动参数。

    Attributes:
        degraded_fraction: 降速状态占总时间的稳态比例，取值 (0, 1)。
        mean_degraded_ms: 降速段平均持续时间（ms），各段时长服从指数分布。
        degraded_bandwidth_factor: 降速状态下的带宽系数。
        degraded_extra_latency_ms: 降速状态下发起读取的额外固定延迟（ms）。
        jitter_cv: 单次读取带宽系数的变异系数（对数正态，均值 1）。
    """

    degraded_fraction: float
    mean_degraded_ms: float
    degraded_bandwidth_factor: float
    degraded_extra_latency_ms: float
    jitter_cv: float


# L3 占位参数：参考 SSD 后台 GC/写入干扰造成的成段慢读，数值未经实测校准。
L3_FLUCTUATION = Fluctuation(
    degraded_fraction=0.02,
    mean_degraded_ms=50.0,
    degraded_bandwidth_factor=0.3,
    degraded_extra_latency_ms=1.0,
    jitter_cv=0.02,
)


@dataclass(frozen=True, slots=True)
class KvSourceSpec:
    """来源的不可变规格；``fluctuation`` 为 None 时按固定延迟与带宽确定性计时。"""

    bytes_per_ms: float
    latency_ms: float
    fluctuation: Fluctuation | None = None


class KvSource:
    """单个来源的链路占用与波动状态；读取开始时刻单调不减，状态序列随时间按需生成。"""

    __slots__ = (
        "_bytes_per_ms",
        "_degraded",
        "_degraded_factor",
        "_degraded_latency_ms",
        "_fluctuating",
        "_jitter_mu",
        "_jitter_sigma",
        "_latency_ms",
        "_mean_degraded_ms",
        "_mean_normal_ms",
        "_rng",
        "_state_end",
        "free_ms",
    )

    def __init__(self, spec: KvSourceSpec, seed: str) -> None:  # 来源规格与种子 → 空闲来源
        self._bytes_per_ms = spec.bytes_per_ms
        self._latency_ms = spec.latency_ms
        self.free_ms = 0.0  # 链路空闲时刻
        self._rng = random.Random(seed)
        self._fluctuating = spec.fluctuation is not None
        self._degraded = False
        self._state_end = math.inf
        self._mean_degraded_ms = self._mean_normal_ms = math.inf
        self._degraded_factor = 1.0
        self._degraded_latency_ms = self._jitter_mu = self._jitter_sigma = 0.0
        fluctuation = spec.fluctuation
        if fluctuation is not None:
            fraction = fluctuation.degraded_fraction
            self._mean_degraded_ms = fluctuation.mean_degraded_ms
            self._mean_normal_ms = fluctuation.mean_degraded_ms * (1 - fraction) / fraction
            self._degraded_factor = fluctuation.degraded_bandwidth_factor
            self._degraded_latency_ms = fluctuation.degraded_extra_latency_ms
            variance = math.log1p(fluctuation.jitter_cv**2)
            self._jitter_sigma = math.sqrt(variance)
            self._jitter_mu = -variance / 2
            self._degraded = self._rng.random() < fraction  # 自稳态分布开始
            self._state_end = self._rng.expovariate(1 / self._mean_state_ms())

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

    def _mean_state_ms(self) -> float:  # 当前状态 → 该状态段的平均时长（ms）
        return self._mean_degraded_ms if self._degraded else self._mean_normal_ms

    def _advance(self, now: float, *, idle: bool = False) -> None:
        if now < self._state_end:
            return
        if idle:
            normal_rate = 1 / self._mean_normal_ms
            degraded_rate = 1 / self._mean_degraded_ms
            stationary = normal_rate / (normal_rate + degraded_rate)
            elapsed = now - self._state_end
            # 已知首次切换时刻；之后空档按两状态链的精确转移概率直接采样。
            probability = stationary + (int(not self._degraded) - stationary) * math.exp(
                -(normal_rate + degraded_rate) * elapsed
            )
            self._degraded = self._rng.random() < probability
            self._state_end = now + self._rng.expovariate(1 / self._mean_state_ms())
        else:
            self._degraded = not self._degraded
            self._state_end += self._rng.expovariate(1 / self._mean_state_ms())

    def _read(self, now: float, size: float) -> float:  # 开始时刻与字节数 → 波动下的完成时刻
        self._advance(now, idle=True)
        now += self._latency_ms + (self._degraded_latency_ms if self._degraded else 0.0)
        self._advance(now, idle=True)
        rate = self._bytes_per_ms * self._rng.lognormvariate(self._jitter_mu, self._jitter_sigma)
        remaining = size
        while True:  # 跨越状态切换时分段按各段带宽传输
            self._advance(now)
            current = rate * self._degraded_factor if self._degraded else rate
            capacity = current * (self._state_end - now)
            if remaining <= capacity:
                return now + remaining / current
            remaining -= capacity
            now = self._state_end
