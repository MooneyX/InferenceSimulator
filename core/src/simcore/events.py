"""事件类型定义与事件流记录。

两层概念，刻意分开：

- **调度事件** ``Event``：压入协调器事件队列的最小载荷，只携带排序键与引用 ID。
  事件队列本身由 ``coordinator`` 持有，本模块只定义事件形态与全序。
- **事件流** ``TraceRecord``：引擎处理事件后产出的自描述记录，按处理顺序追加到
  ``EventLog``。内核产出完整事件流但不经 API 暴露，为决策溯源与播放式仿真预留。

同刻全序（可复现性的前提）：``(time, kind, ref, seq)``。

- ``kind`` 优先级：步末发布、请求结算及客户端回调先于传输完成，
  传输完成先于本时刻的发送，所有发送先于新引擎步。
  将来才到达的闭环子轮与普通发送同级。
- ``ref`` 在同类事件内按请求 ID 排序；``seq`` 只负责相同 ID 的最终平局。

依赖层级：仅依赖 ``schemas``。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

from simcore.schemas import Arrival

__all__ = [
    "ArrivalTrace",
    "DoneTrace",
    "Event",
    "EventFactory",
    "EventKind",
    "EventLog",
    "StepTrace",
    "TraceRecord",
]


class EventKind(IntEnum):
    """事件类别。枚举值即同刻优先级，越小越先处理。"""

    STEP_DONE = 0
    REQUEST_DONE = 1
    CLIENT_COMPLETION = 2
    L3_LOOKUP_DONE = 3
    L3_L2_DONE = 4
    L2_L1_DONE = 5
    CLIENT_CHILD_SEND = 6
    CLIENT_SEND = 7
    ARRIVAL = 7  # 手写到达事件别名
    OPEN_LOOP_TICK = 8
    ENGINE_STEP = 9


@dataclass(frozen=True, slots=True, order=True)
class Event:
    """调度事件。按 ``(time, kind, ref, seq)`` 排序，可直接入 ``heapq``。

    同一时间同类事件优先处理较小的请求 ID，序号仅打破相同 ID 的平局。

    Attributes:
        time: 事件触发时刻（ms）。
        kind: 事件类别。
        ref: 发送为 turn_id；完成/回调为 request_id；引擎步为 instance_id。
        seq: 创建序号，由 ``EventFactory`` 分配。
    """

    time: float
    kind: EventKind
    ref: int
    seq: int


class EventFactory:
    """事件构造器：唯一的 ``seq`` 分配点，保证序号单调且不重复。"""

    __slots__ = ("_next_seq",)

    def __init__(self) -> None:
        self._next_seq = 0

    def _make(self, time: float, kind: EventKind, ref: int) -> Event:
        seq = self._next_seq
        self._next_seq = seq + 1
        return Event(time, kind, ref, seq)

    def arrival(self, time: float, request_id: int) -> Event:
        """旧单轮到达事件；映射到客户端发送的同刻优先级。"""
        return self._make(time, EventKind.ARRIVAL, request_id)

    def client_completion(self, time: float, request_id: int) -> Event:
        return self._make(time, EventKind.CLIENT_COMPLETION, request_id)

    def client_send(self, time: float, turn_id: int) -> Event:
        return self._make(time, EventKind.CLIENT_SEND, turn_id)

    def client_child_send(self, time: float, turn_id: int) -> Event:
        """完成回调在本时刻唤醒的子轮须先于其他普通发送。"""
        return self._make(time, EventKind.CLIENT_CHILD_SEND, turn_id)

    def open_loop_tick(self, time: float, index: int) -> Event:
        return self._make(time, EventKind.OPEN_LOOP_TICK, index)

    def engine_step(self, time: float, instance_id: int) -> Event:
        return self._make(time, EventKind.ENGINE_STEP, instance_id)

    def step_done(self, time: float, instance_id: int) -> Event:
        return self._make(time, EventKind.STEP_DONE, instance_id)

    def l3_lookup_done(self, time: float, request_id: int) -> Event:
        return self._make(time, EventKind.L3_LOOKUP_DONE, request_id)

    def l3_l2_done(self, time: float, transfer_id: int) -> Event:
        return self._make(time, EventKind.L3_L2_DONE, transfer_id)

    def l2_l1_done(self, time: float, transfer_id: int) -> Event:
        return self._make(time, EventKind.L2_L1_DONE, transfer_id)

    def request_done(self, time: float, request_id: int) -> Event:
        return self._make(time, EventKind.REQUEST_DONE, request_id)

    @property
    def issued(self) -> int:
        """已分配的事件数。"""
        return self._next_seq


# ---------------------------------------------------------------- 事件流记录


@dataclass(frozen=True, slots=True)
class ArrivalTrace:
    """请求到达并进入等待队列。"""

    time: float
    request_id: int
    prompt_len: int
    output_len: int

    @classmethod
    def of(cls, arrival: Arrival) -> "ArrivalTrace":
        return cls(arrival.sim_time, arrival.request_id, arrival.prompt_len, arrival.output_len)


@dataclass(frozen=True, slots=True)
class StepTrace:
    """一个引擎步的完整决策与结果，是决策溯源的主要载体。

    Attributes:
        time: 步起点（ms）。
        step_index: 步序号，从 0 开始连续编号。
        duration: 步长（ms）。
        admitted: 本步准入的 request_id，按准入顺序。
        forced: 其中强制准入的 request_id。
        num_decoding: 本步 decode 的请求数。
        batch_total_tokens: 本步 batch 的 KV token 总量（含本步准入 prompt）。
        kv_used: 步末 decode 增长后、释放完成请求前的 KV 占用。
        kv_overflow: 步末超出预算的 token 数；未超限为 0。
        finished: 本步结束时完成的 request_id。
    """

    time: float
    step_index: int
    duration: float
    admitted: tuple[int, ...]
    forced: tuple[int, ...]
    num_decoding: int
    batch_total_tokens: int
    kv_used: int
    kv_overflow: int
    finished: tuple[int, ...]
    probes: int = 0
    matched_cache_tokens: int = 0
    actual_hit_tokens: int = 0
    allocated_blocks: int = 0
    evicted_blocks: int = 0
    l1_active_before: int = 0
    l1_evictable_before: int = 0
    l1_free_before: int = 0
    l1_active_after: int = 0
    l1_evictable_after: int = 0
    l1_free_after: int = 0
    l1_internal_fragmentation_tokens: int = 0
    phase: Literal["PREFILL", "DECODE", "MIXED", "RESTORE"] | None = None
    withdrawn: tuple[int, ...] = ()
    aborted: tuple[int, ...] = ()
    restored: tuple[int, ...] = ()
    restore_sources: tuple[tuple[str, str], ...] = ()
    restore_bytes: int = 0
    restore_time_ms: float = 0.0
    l1_evictions: int = 0
    l2_evictions: int = 0
    l3_evictions: int = 0
    promotions: int = 0
    demotions: int = 0

    @property
    def end(self) -> float:
        return self.time + self.duration


@dataclass(frozen=True, slots=True)
class DoneTrace:
    """请求完成结算，KV 额度已归还。"""

    time: float
    request_id: int
    released_tokens: int


TraceRecord = ArrivalTrace | StepTrace | DoneTrace


class EventLog:
    """按处理顺序追加的事件流。

    不变式：记录时刻单调不减——违反说明引擎的事件处理顺序有误，立即抛出。

    ``enabled=False`` 时 ``record`` 为空操作（仍做时序校验），供基准测试剥离记录开销。
    """

    __slots__ = ("_enabled", "_last_time", "_records")

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._records: list[TraceRecord] = []
        self._last_time = float("-inf")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def record(self, entry: TraceRecord) -> None:
        """追加一条记录。

        Raises:
            ValueError: 记录时刻早于上一条。
        """
        if entry.time < self._last_time:
            raise ValueError(
                f"事件流时刻倒退：{type(entry).__name__}@{entry.time} 早于上一条 @{self._last_time}"
            )
        self._last_time = entry.time
        if self._enabled:
            self._records.append(entry)

    def __iter__(self) -> Iterator[TraceRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def steps(self) -> Iterator[StepTrace]:
        return (r for r in self._records if isinstance(r, StepTrace))

    def for_request(self, request_id: int) -> list[TraceRecord]:
        """某请求相关的全部记录：到达、参与的步（准入或完成）、完成结算。"""
        out: list[TraceRecord] = []
        for r in self._records:
            match r:
                case ArrivalTrace() | DoneTrace() if r.request_id == request_id:
                    out.append(r)
                case StepTrace() if (
                    request_id in r.admitted or request_id in r.restored or request_id in r.finished
                ):
                    out.append(r)
        return out
