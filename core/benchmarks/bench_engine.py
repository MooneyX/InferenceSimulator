"""内核基准测试：固定种子、固定规模的计时回归用例（M1 起常驻）。

本基准用例的验收线：

- 同 seed 两次运行结果逐字节一致。
- 100/1k/10k 两种 P/D 模式报告墙钟、步数及物理页驻留峰值。

不设与机器性能绑定的硬耗时阈值。

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
    LognormalLength,
    PoissonLength,
    SimConfig,
    SimResult,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    simulate,
)
from simcore.schemas import (
    EmpiricalTimingConfig,
    HardwareSpec,
    ModelSpec,
    PhysicalInferenceConfig,
    SchedulerConfig,
    StorageSpec,
)

BENCHMARK_SIZES = (100, 1_000, 10_000)
REPEATS = 3


def bench_config(num_requests: int, seed: int = 0, *, exclusive: bool = True) -> SimConfig:
    """基准负载：长尾 prompt + 泊松 output。

    到达率（50 req/s）约为服务能力（约 5.6 req/s）的 9 倍，刻意过载：batch 常驻满载、
    等待队列持续增长，覆盖最重的调度路径。

    固定不变——修改会使历史耗时不可比。
    """
    return SimConfig(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=50.0,
            seed=seed,
            synthetic=SyntheticWorkloadConfig(
                num_requests=num_requests,
                prompt_len=LognormalLength(mean=512.0, sigma=1.0),
                turn_increment_len=LognormalLength(mean=16.0, sigma=1.0),
                output_len=PoissonLength(mean=128.0),
            )
        ),
        scheduler=SchedulerConfig(max_running_requests=32, pd_mutually_exclusive=exclusive),
        inference=PhysicalInferenceConfig(
            page_size=64,
            model=ModelSpec(name="toy", num_layers=2, hidden_size=16,
                            num_attention_heads=2, num_kv_heads=1, head_dim=8,
                            ffn_intermediate_size=32, vocab_size=16),
            hardware=HardwareSpec(name="toy", memory_bytes_per_gpu=10_000_000,
                                  effective_fp16_tflops=1, effective_hbm_gb_s=1),
            storage=StorageSpec(l1_available_gb=(10_464 + 65_536 * 64) / 1_000_000_000),
            timing=EmpiricalTimingConfig(
                prefill_c1=0.05, prefill_c2=1e-5, decode_d1=8.0, decode_d2=0.002,
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class Timing:
    num_requests: int
    exclusive: bool
    num_engine_steps: int
    peak_resident_pages: int
    best_seconds: float
    median_seconds: float

    def render(self) -> str:
        mode = "exclusive" if self.exclusive else "fused"
        return (
            f"{self.num_requests:>7,} req  {mode:>9}  "
            f"best {self.best_seconds * 1000:9.1f} ms  "
            f"median {self.median_seconds * 1000:9.1f} ms  "
            f"steps {self.num_engine_steps:>8,}  peak KV {self.peak_resident_pages:>5,} pages"
        )


def measure(
    fn: Callable[[], SimResult], repeats: int = REPEATS
) -> tuple[SimResult, list[float]]:
    """运行 ``repeats`` 次，返回最后一次结果与每次耗时（秒）。"""
    durations: list[float] = []
    result: SimResult | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    assert result is not None
    return result, durations


def time_simulate(num_requests: int, *, exclusive: bool, repeats: int = REPEATS) -> Timing:
    cfg = bench_config(num_requests, exclusive=exclusive)
    result, durations = measure(lambda: simulate(cfg), repeats)
    ordered = sorted(durations)
    return Timing(
        num_requests=num_requests,
        exclusive=exclusive,
        num_engine_steps=result.meta.num_engine_steps,
        peak_resident_pages=result.metrics.l1_peak_resident_blocks,
        best_seconds=ordered[0],
        median_seconds=ordered[len(ordered) // 2],
    )


# ---------------------------------------------------------------- pytest 入口

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("num_requests", BENCHMARK_SIZES)
@pytest.mark.parametrize("exclusive", [True, False], ids=["exclusive", "fused"])
def test_scale_report(num_requests: int, exclusive: bool) -> None:
    timing = time_simulate(num_requests, exclusive=exclusive)
    print(f"\n{timing.render()}")
    assert timing.num_engine_steps > 0
    assert 0 <= timing.peak_resident_pages <= 65_536 // 64


@pytest.mark.parametrize("exclusive", [True, False], ids=["exclusive", "fused"])
def test_acceptance_same_seed_byte_identical(exclusive: bool) -> None:
    cfg = bench_config(100, seed=42, exclusive=exclusive)
    exclude = {"meta": {"wall_clock_ms"}}
    first = simulate(cfg)
    second = simulate(cfg)
    assert first.model_dump_json(exclude=exclude) == second.model_dump_json(exclude=exclude)


# ---------------------------------------------------------------- 脚本入口


def main() -> int:
    print(f"simcore benchmark (python {sys.version.split()[0]}, best of {REPEATS})")
    for num_requests in BENCHMARK_SIZES:
        for exclusive in (True, False):
            print(time_simulate(num_requests, exclusive=exclusive).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
