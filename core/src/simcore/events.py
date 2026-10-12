"""调度事件类型与同刻全序。

``Event`` 是压入协调器事件队列的最小载荷，只携带排序键与引用 ID。
事件队列本身由 ``coordinator`` 持有。

同刻全序（可复现性的前提）：``(time, kind, ref, seq)``。

- ``kind`` 优先级：步末发布、请求结算及客户端回调先于传输完成，
  传输完成先于本时刻的发送，所有发送先于新引擎步。
  将来才到达的闭环子轮与普通发送同级。
- ``ref`` 在同类事件内排序（请求、实例、传输或时钟序号）；``seq`` 打破同引用的平局。

依赖层级：仅依赖标准库。
"""

from dataclasses import dataclass
from enum import IntEnum

__all__ = ["Event", "EventFactory", "EventKind"]


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
    OPEN_LOOP_TICK = 8
    ENGINE_STEP = 9


@dataclass(frozen=True, slots=True, order=True)
class Event:
    """调度事件。按 ``(time, kind, ref, seq)`` 排序，可直接入 ``heapq``。

    同一时间同类事件按引用 ID 排序；引用可指请求、副本、传输或时钟序号。

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

    def __init__(self) -> None:  # 无显式输入 → 初始化单调事件序号；无返回值
        self._next_seq = 0

    def _make(self, time: float, kind: EventKind, ref: int) -> Event:  # 时刻、类别、引用 → 唯一序号事件
        seq = self._next_seq
        self._next_seq = seq + 1
        return Event(time, kind, ref, seq)

    def client_completion(self, time: float, request_id: int) -> Event:  # 完成时刻与请求 ID → 客户端回调事件
        return self._make(time, EventKind.CLIENT_COMPLETION, request_id)

    def client_send(self, time: float, turn_id: int) -> Event:  # 发送时刻与轮次 ID → 普通发送事件
        return self._make(time, EventKind.CLIENT_SEND, turn_id)

    def client_child_send(self, time: float, turn_id: int) -> Event:  # 回调时刻与子轮 ID → 优先发送事件
        """完成回调在本时刻唤醒的子轮须先于其他普通发送。"""
        return self._make(time, EventKind.CLIENT_CHILD_SEND, turn_id)

    def open_loop_tick(self, time: float, index: int) -> Event:  # 投放时刻与序号 → 开环时钟事件
        return self._make(time, EventKind.OPEN_LOOP_TICK, index)

    def engine_step(self, time: float, instance_id: int) -> Event:  # 步起点与副本 ID → 引擎步事件
        return self._make(time, EventKind.ENGINE_STEP, instance_id)

    def step_done(self, time: float, instance_id: int) -> Event:  # 步终点与副本 ID → 步完成事件
        return self._make(time, EventKind.STEP_DONE, instance_id)

    def l3_lookup_done(self, time: float, request_id: int) -> Event:  # 查询完成时刻与请求 ID → L3 查询事件
        return self._make(time, EventKind.L3_LOOKUP_DONE, request_id)

    def l3_l2_done(self, time: float, transfer_id: int) -> Event:  # 完成时刻与传输 ID → L3 上行事件
        return self._make(time, EventKind.L3_L2_DONE, transfer_id)

    def l2_l1_done(self, time: float, transfer_id: int) -> Event:  # 完成时刻与传输 ID → L2 上行事件
        return self._make(time, EventKind.L2_L1_DONE, transfer_id)

    def request_done(self, time: float, request_id: int) -> Event:  # 结算时刻与请求 ID → 完成事件
        return self._make(time, EventKind.REQUEST_DONE, request_id)
