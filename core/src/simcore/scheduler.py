"""FCFS 调度器：决定每个 ``EngineStep`` 的 batch 组成与步长。

职责边界：
- 持有等待队列，按到达顺序准入；准入时直接在 KV 账本上占用额度（决策即提交）。
- 依据性能模型给出本步步长。
- 不感知仿真时钟与事件队列：何时调用、步长如何推进时钟、请求何时完成，均由引擎负责。

调度规则（见 ``docs/design/kernel.md``）：
- 严格 FCFS：队头放不下即停止本步准入，不跳过队头。
- 双重约束：batch 内请求数 <= ``max_batch_size``，且准入不使 KV 占用超预算。
- 超大请求强制准入：队头 prompt 超过整个预算时，仅当 batch 为空才强制准入，避免死锁。

与引擎的约定：batch = 全部 KV 驻留请求；引擎须在步末释放完成请求，
故每次 ``plan_step`` 开始时驻留请求均处于 decode 阶段。

依赖层级：依赖 ``schemas``、``kv_cache``、``perf_model``。
"""

from collections import deque
from dataclasses import dataclass

from simcore.kv_cache import KVCache
from simcore.perf_model import step_time
from simcore.schemas import Arrival, EngineConfig, PerfModelConfig

__all__ = ["FCFSScheduler", "StepPlan"]


@dataclass(frozen=True, slots=True)
class StepPlan:
    """一个引擎步的调度结果。

    Attributes:
        admitted: 本步新准入的请求，按准入顺序（即到达顺序）。
        forced: 其中越过预算强制准入的请求 ID。
        num_decoding: 本步执行 decode 的请求数（= 步开始时的驻留请求数）。
        batch_total_tokens: 本步 batch 的 KV token 总量（含本步准入请求的 prompt）。
        duration: 本步步长（ms），融合批语义。
    """

    admitted: tuple[Arrival, ...]
    forced: tuple[int, ...]
    num_decoding: int
    batch_total_tokens: int
    duration: float

    @property
    def has_decode(self) -> bool:
        return self.num_decoding > 0

    @property
    def batch_size(self) -> int:
        return self.num_decoding + len(self.admitted)


class FCFSScheduler:
    """先来先服务调度器。"""

    __slots__ = ("_kv", "_last_enqueued_time", "_max_batch_size", "_perf", "_waiting")

    def __init__(self, engine: EngineConfig, perf: PerfModelConfig, kv: KVCache) -> None:
        if kv.budget != engine.kv_token_budget:
            raise ValueError(
                f"KV 账本预算 {kv.budget} 与配置 kv_token_budget {engine.kv_token_budget} 不一致"
            )
        self._max_batch_size = engine.max_batch_size
        self._perf = perf
        self._kv = kv
        self._waiting: deque[Arrival] = deque()
        self._last_enqueued_time = float("-inf")

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    def enqueue(self, arrival: Arrival) -> None:
        """请求到达，加入等待队列尾部。

        Raises:
            ValueError: 到达时刻早于上一个入队请求——FCFS 顺序依赖到达升序。
        """
        if arrival.sim_time < self._last_enqueued_time:
            raise ValueError(
                f"请求 {arrival.request_id} 到达时刻 {arrival.sim_time} "
                f"早于上一入队请求 {self._last_enqueued_time}，违反到达升序"
            )
        self._last_enqueued_time = arrival.sim_time
        self._waiting.append(arrival)

    def plan_step(self) -> StepPlan | None:
        """规划下一个引擎步：执行准入并计算步长。

        Returns:
            本步计划；既无驻留请求也无等待请求时返回 ``None``（引擎应跳跃时钟）。

        进展保证：batch 为空且有等待请求时，队头必被准入（正常或强制），
        因此只要有请求在等待，引擎就不会陷入空转。
        """
        kv = self._kv
        waiting = self._waiting
        num_decoding = len(kv)
        if num_decoding == 0 and not waiting:
            return None

        capacity = self._max_batch_size - num_decoding
        admitted: list[Arrival] = []
        forced: list[int] = []
        while waiting and len(admitted) < capacity:
            head = waiting[0]
            if kv.can_admit(head.prompt_len):
                kv.reserve(head.request_id, head.prompt_len)
            elif len(kv) == 0 and not kv.can_ever_admit(head.prompt_len):
                kv.reserve(head.request_id, head.prompt_len, force=True)
                forced.append(head.request_id)
            else:
                break
            waiting.popleft()
            admitted.append(head)

        batch_total_tokens = kv.used
        duration = step_time(
            [a.prompt_len for a in admitted],
            batch_total_tokens,
            self._perf,
            has_decode=num_decoding > 0,
        )
        return StepPlan(
            admitted=tuple(admitted),
            forced=tuple(forced),
            num_decoding=num_decoding,
            batch_total_tokens=batch_total_tokens,
            duration=duration,
        )
