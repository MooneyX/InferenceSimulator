"""固定合成 session 场景的端到端耗时与 Python 峰值分配对照。

同一脚本可分别加载当前版本与旧版 ``simcore``（通过 PYTHONPATH 指定源码），
比较之前须确认两个版本报告的 step、理想/实际命中和 Prefill 数相同。
计时与 tracemalloc 分开运行；峰值只表示 Python 跟踪到的分配，不是进程 RSS。

    python core/benchmarks/bench_chunk_identity.py --requests 120 --sessions 30
"""

import argparse
import gc
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import simcore
from simcore import (
    FixedLength,
    SessionConfig,
    SimConfig,
    SyntheticWorkloadConfig,
    WorkloadConfig,
    simulate,
)
from simcore.schemas import (
    EmpiricalTimingConfig,
    HardwareSpec,
    InferenceConfig,
    ModelSpec,
    SchedulerConfig,
    StorageSpec,
)


def bench_config(requests: int, sessions: int, block_size: int) -> SimConfig:
    return SimConfig(
        num_instances=1,
        workload=WorkloadConfig(
            session_rate=80.0,
            seed=17,
            synthetic=SyntheticWorkloadConfig(
                num_requests=requests,
                prompt_len=FixedLength(value=64),
                turn_increment_len=FixedLength(value=1),
                output_len=FixedLength(value=16),
                session=SessionConfig(
                    num_sessions=sessions,
                    think_time_ms=5.0,
                ),
            ),
        ),
        scheduler=SchedulerConfig(max_running_requests=8),
        inference=InferenceConfig(
            page_size=block_size,
            model=ModelSpec(
                name="toy",
                num_layers=2,
                hidden_size=16,
                num_attention_heads=2,
                num_kv_heads=1,
                head_dim=8,
                ffn_intermediate_size=32,
                vocab_size=16,
            ),
            hardware=HardwareSpec(
                name="toy",
                memory_bytes_per_gpu=10_464 + 4096 * 64,
                effective_fp16_tflops=1,
                effective_hbm_gb_s=1,
            ),
            storage=StorageSpec(),
            timing=EmpiricalTimingConfig(prefill_c1=0.05, decode_d1=2.0, decode_d2=0.002),
        ),
    )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=positive_int, default=120)
    parser.add_argument("--sessions", type=positive_int, default=30)
    parser.add_argument("--block-size", type=positive_int, default=16)
    parser.add_argument("--repeats", type=positive_int, default=3)
    args = parser.parse_args()
    if args.sessions > args.requests:
        parser.error("sessions cannot exceed requests")
    config = bench_config(args.requests, args.sessions, args.block_size)
    result = simulate(config)
    exclude = {"meta": {"wall_clock_ms"}}
    warmup = result.model_dump_json(exclude=exclude)
    durations: list[float] = []
    for _ in range(args.repeats):
        gc.collect()
        start = time.perf_counter()
        result = simulate(config)
        durations.append(time.perf_counter() - start)
    if warmup != result.model_dump_json(exclude=exclude):
        raise RuntimeError("same seed produced different simulation results")
    del warmup
    del result
    gc.collect()

    tracemalloc.start()
    try:
        result = simulate(config)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    metrics = result.metrics
    print(f"python={sys.version.split()[0]} simcore={Path(simcore.__file__).resolve()}")
    print(
        f"requests={args.requests} sessions={args.sessions} block_size={args.block_size} "
        f"seed=17 repeats={args.repeats}"
    )
    print(
        f"best_s={min(durations):.6f} median_s={statistics.median(durations):.6f} "
        f"python_peak_mib={peak / 1024**2:.3f}"
    )
    print(
        f"steps={result.meta.num_engine_steps} total_input_tokens={metrics.total_input_tokens} "
        f"ideal_hit_tokens={metrics.ideal_hit_tokens} "
        f"actual_hit_tokens={metrics.actual_hit_tokens} "
        f"actual_prefill_tokens={metrics.actual_prefill_tokens} "
        f"l1_evictions={metrics.l1_evictions}"
    )


if __name__ == "__main__":
    main()
