"""闭环客户端：只在父请求实际完成后创建下一轮发送。"""

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    InferenceRequest,
    RequestCompletion,
    WorkloadPlan,
)

__all__ = ["ClientDriver"]


class ClientDriver:
    """根据已完成父轮的请求身份和实际生成数构造继承前缀。"""

    def __init__(self, plan: WorkloadPlan, engine: EngineConfig) -> None:
        if plan.block_size_tokens != engine.page_size:
            raise ValueError("计划和引擎 page_size 不一致")
        self.plan = plan
        self.page_size = engine.page_size
        self._ready: dict[int, float] = {}
        self._inflight: dict[int, InferenceRequest] = {}
        self._finished: dict[int, tuple[InferenceRequest, int]] = {}
        self._sent: set[int] = set()
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
        inherited = turn.ideal_reuse_blocks
        if inherited > (turn.prompt_len - 1) // self.page_size:
            raise ValueError("继承页超过子轮输入的完整前缀")
        if turn.parent_request_id is not None:
            parent = self._finished.get(turn.parent_request_id)
            if parent is None or parent[1] // self.page_size < inherited:
                raise ValueError("子轮次需要父请求完成后的完整块上下文")
            head = tuple(
                logical_chunk_id(parent[0].request_id, index, parent[0].inherited_prefix_ids)
                for index in range(inherited)
            )
        else:
            if inherited:
                raise ValueError("根请求不能继承完整块")
            head = ()
        request = InferenceRequest(
            request_id=turn.request_id,
            session_id=turn.session_id,
            parent_request_id=turn.parent_request_id,
            prompt_len=turn.prompt_len,
            max_new_tokens=turn.max_new_tokens,
            inherited_prefix_ids=head,
            ideal_reuse_blocks=inherited,
        )
        self._sent.add(turn.request_id)
        self._inflight[turn.request_id] = request
        del self._ready[turn.request_id]
        return request

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:
        request = self._inflight.get(completion.request_id)
        if request is None:
            raise ValueError("完成了未发送或已完成的请求")
        if not 1 <= completion.generated_tokens <= request.max_new_tokens:
            raise ValueError("完成的生成数超出请求可见上限")
        self._finished[request.request_id] = (
            request,
            request.prompt_len + completion.generated_tokens - 1,
        )
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
