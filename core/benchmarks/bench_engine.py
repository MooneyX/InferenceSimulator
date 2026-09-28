"""内核基准测试：固定种子、固定规模的计时回归用例（M1 起常驻）。

验收线（见 ``docs/design/roadmap.md``）：

- 1k 请求全场仿真 < 1s（取多次运行的最小值，排除调度抖动）。
- 同种子两次运行结果逐字节一致。

10k 规模只报告耗时、不设阈值，用于观察扩展趋势。

运行方式（默认 ``pytest`` 不收集本目录，避免拖慢常规单测）::

    python -m pytest core/benchmarks        # 断言验收线
    python core/benchmarks/bench_engine.py  # 打印耗时报告

计时代码只存在于本目录；``simcore`` 自身不读取时钟，保持零 IO。
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from simcore import (
    EngineConfig,
    LognormalLength,
    PerfModelConfig,
    PoissonLength,
    SimConfig,
    SimResult,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    simulate,
)

ACCEPTANCE_REQUESTS = 1_000
ACCEPTANCE_SECONDS = 1.0
SCALE_REQUESTS = 10_000
REPEATS = 5


def bench_config(num_requests: int, seed: int = 0) -> SimConfig:
    """基准负载：长尾 prompt + 泊松 output。

    到达率（50 req/s）约为服务能力（约 5.6 req/s）的 9 倍，刻意过载：batch 常驻满载、
    等待队列持续增长，覆盖最重的调度路径。

    固定不变——修改会使历史耗时不可比。
    """
    return SimConfig(
        workload=WorkloadConfig(
            synthetic=SyntheticWorkloadConfig(
                num_requests=num_requests,
                arrival_rate=50.0,
                prompt_len=LognormalLength(mean=512.0, sigma=1.0),
                output_len=PoissonLength(mean=128.0),
                seed=seed,
            )
        ),
        engine=EngineConfig(max_batch_size=32, kv_token_budget=65_536),
        perf_model=PerfModelConfig(
            prefill_c1=0.05, prefill_c2=1e-5, decode_d1=8.0, decode_d2=0.002
        ),
    )


@dataclass(frozen=True, slots=True)
class Timing:
    num_requests: int
    num_engine_steps: int
    best_seconds: float
    median_seconds: float

    @property
    def requests_per_second(self) -> float:
        return self.num_requests / self.best_seconds

    def render(self) -> str:
        best_ms = self.best_seconds * 1000
        median_ms = self.median_seconds * 1000
        return (
            f"{self.num_requests:>7,} req  {self.num_engine_steps:>8,} steps  "
            f"best {best_ms:8.1f} ms  median {median_ms:8.1f} ms  "
            f"{self.requests_per_second:>10,.0f} req/s"
        )


def measure(fn: Callable[[], SimResult], repeats: int = REPEATS) -> tuple[SimResult, list[float]]:
    """运行 ``repeats`` 次，返回最后一次结果与每次耗时（秒）。"""
    durations: list[float] = []
    result: SimResult | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    assert result is not None
    return result, durations


def time_simulate(num_requests: int, repeats: int = REPEATS) -> Timing:
    cfg = bench_config(num_requests)
    result, durations = measure(lambda: simulate(cfg), repeats)
    ordered = sorted(durations)
    return Timing(
        num_requests=num_requests,
        num_engine_steps=result.meta.num_engine_steps,
        best_seconds=ordered[0],
        median_seconds=ordered[len(ordered) // 2],
    )


# ---------------------------------------------------------------- pytest 入口

pytestmark = pytest.mark.benchmark


def test_acceptance_1k_under_one_second() -> None:
    timing = time_simulate(ACCEPTANCE_REQUESTS)
    print(f"\n{timing.render()}")
    assert timing.best_seconds < ACCEPTANCE_SECONDS, (
        f"1k 请求耗时 {timing.best_seconds:.3f}s 超出验收线 {ACCEPTANCE_SECONDS}s"
    )


def test_acceptance_same_seed_byte_identical() -> None:
    cfg = bench_config(ACCEPTANCE_REQUESTS, seed=42)
    assert simulate(cfg).model_dump_json() == simulate(cfg).model_dump_json()


def test_scale_10k_report() -> None:
    """仅报告，不设阈值。"""
    timing = time_simulate(SCALE_REQUESTS, repeats=3)
    print(f"\n{timing.render()}")
    assert timing.num_engine_steps > 0


# ---------------------------------------------------------------- 脚本入口


def main() -> int:
    print(f"simcore benchmark (python {sys.version.split()[0]}, best of {REPEATS})")
    acceptance = time_simulate(ACCEPTANCE_REQUESTS)
    print(acceptance.render())
    print(time_simulate(SCALE_REQUESTS, repeats=3).render())
    passed = acceptance.best_seconds < ACCEPTANCE_SECONDS
    print(f"acceptance (1k < {ACCEPTANCE_SECONDS}s): {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
