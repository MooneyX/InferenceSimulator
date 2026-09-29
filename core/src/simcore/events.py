"""事件类型定义与事件流记录。

两层概念，刻意分开：

- **调度事件** ``Event``：压入协调器事件队列的最小载荷，只携带排序键与引用 ID。
  事件队列本身由 ``coordinator`` 持有，本模块只定义事件形态与全序。
- **事件流** ``TraceRecord``：引擎处理事件后产出的自描述记录，按处理顺序追加到
  ``EventLog``。内核产出完整事件流但不经 API 暴露，为决策溯源与播放式仿真预留。

同刻全序（可复现性的前提）：``(time, kind, seq)``。

- ``kind`` 优先级：``REQUEST_DONE`` < ``CLIENT_COMPLETION`` < ``CLIENT_SEND``
  < ``ENGINE_STEP``。同一时刻先结算全部完成并释放 KV，再通知客户端，接着发送
  已就绪的请求，最后执行引擎步，使零思考时间的子请求能看到父请求释放的资源。
- ``seq`` 由 ``EventFactory`` 单调分配，打破同刻同类事件的平局，等价于按创建顺序。

依赖层级：仅依赖 ``schemas``。
"""

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum

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

    REQUEST_DONE = 0
    CLIENT_COMPLETION = 1
    CLIENT_SEND = 2
    ARRIVAL = 2  # 旧调用方的发送事件别名
    ENGINE_STEP = 3


@dataclass(frozen=True, slots=True, order=True)
class Event:
    """调度事件。按字段声明顺序 ``(time, kind, seq, ref)`` 比较，可直接入 ``heapq``。

    ``seq`` 全局唯一，故比较永远不会落到 ``ref``。

    Attributes:
        time: 事件触发时刻（ms）。
        kind: 事件类别。
        seq: 创建序号，由 ``EventFactory`` 分配。
        ref: ``CLIENT_SEND`` 为 turn_id；完成/回调事件为 request_id；
            ``ENGINE_STEP`` 为步序号。
    """

    time: float
    kind: EventKind
    seq: int
    ref: int


class EventFactory:
    """事件构造器：唯一的 ``seq`` 分配点，保证序号单调且不重复。"""

    __slots__ = ("_next_seq",)

    def __init__(self) -> None:
        self._next_seq = 0

    def _make(self, time: float, kind: EventKind, ref: int) -> Event:
        seq = self._next_seq
        self._next_seq = seq + 1
        return Event(time, kind, seq, ref)

    def arrival(self, time: float, request_id: int) -> Event:
        """旧单轮到达事件；映射到客户端发送的同刻优先级。"""
        return self._make(time, EventKind.ARRIVAL, request_id)

    def client_completion(self, time: float, request_id: int) -> Event:
        return self._make(time, EventKind.CLIENT_COMPLETION, request_id)

    def client_send(self, time: float, turn_id: int) -> Event:
        return self._make(time, EventKind.CLIENT_SEND, turn_id)

    def engine_step(self, time: float, step_index: int) -> Event:
        return self._make(time, EventKind.ENGINE_STEP, step_index)

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
                case StepTrace() if request_id in r.admitted or request_id in r.finished:
                    out.append(r)
        return out
