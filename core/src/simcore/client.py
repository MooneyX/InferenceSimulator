"""闭环客户端：只在父请求实际完成后创建下一轮发送。"""

from simcore.schemas import (
    ClientWakeup,
    InferenceRequest,
    PrefixCacheConfig,
    RequestCompletion,
    WorkloadPlan,
)
from simcore.workload import make_request

__all__ = ["ClientDriver"]


class ClientDriver:
    """持有 session 逻辑上下文、待发送轮次及唯一公共前缀。"""

    def __init__(self, plan: WorkloadPlan, cache: PrefixCacheConfig) -> None:
        if plan.block_size_tokens != cache.block_size_tokens:
            raise ValueError("计划和缓存 block_size_tokens 不一致")
        self.plan = plan
        self.cache = cache
        self._ready: dict[int, float] = {}
        self._inflight: dict[int, InferenceRequest] = {}
        self._finished: dict[int, tuple[int, ...]] = {}
        self._sent: set[int] = set()
        self._public_prefix: tuple[int, ...] = ()
        for request_id in plan.root_request_ids:
            time = plan.turns[request_id].root_arrival_time_ms
            if time is None:
                raise ValueError("根请求缺少计划到达时间")
            self._ready[request_id] = time

    def initial_wakeups(self) -> tuple[ClientWakeup, ...]:
        return tuple(
            ClientWakeup(self.plan.turns[key].session_id, key, self._ready[key])
            for key in self.plan.root_request_ids
            if key in self._ready
        )

    def on_wakeup(self, wakeup: ClientWakeup) -> InferenceRequest:
        turn = self.plan.turns[wakeup.turn_id]
        if turn.session_id != wakeup.session_id:
            raise ValueError("客户端唤醒的 session_id 与轮次不一致")
        planned = self._ready.get(wakeup.turn_id)
        if planned is None or planned != wakeup.send_time_ms or wakeup.turn_id in self._sent:
            raise ValueError("轮次未就绪、重复发送或发送时间不符")
        context = (
            self._finished[turn.parent_request_id] if turn.parent_request_id is not None else None
        )
        request = make_request(turn, self.cache, context, self._public_prefix)
        if turn.parent_request_id is None and not self._sent:
            self._public_prefix = request.input_token_ids
        self._sent.add(turn.request_id)
        self._inflight[turn.request_id] = request
        del self._ready[turn.request_id]
        return request

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:
        request = self._inflight.get(completion.request_id)
        if request is None:
            raise ValueError("完成了未发送或已完成的请求")
        if len(completion.output_token_ids) != len(request.output_token_plan):
            raise ValueError("实际输出 token 数不符合计划")
        self._finished[request.request_id] = request.input_token_ids + completion.output_token_ids
        del self._inflight[completion.request_id]
        wakeups: list[ClientWakeup] = []
        for child_id in self.plan.children.get(request.request_id, ()):
            child = self.plan.turns[child_id]
            if child.think_time_ms is None:
                raise ValueError("子轮次缺少 think time")
            time = completion.finish_time_ms + child.think_time_ms
            self._ready[child_id] = time
            wakeups.append(ClientWakeup(child.session_id, child_id, time))
        return tuple(wakeups)
