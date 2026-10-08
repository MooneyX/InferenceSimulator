"""闭环客户端：只在父请求实际完成后创建下一轮发送。"""

from simcore.prefix_hash import logical_chunk_id
from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    InferenceRequest,
    RequestCompletion,
    SkippedRequest,
    WorkloadPlan,
)

__all__ = ["ClientDriver"]


class ClientDriver:
    """根据已完成父轮的请求身份和实际生成数构造继承前缀。"""

    def __init__(self, plan: WorkloadPlan, engine: EngineConfig) -> None:
        self.plan = plan
        self.page_size = engine.page_size
        self._inflight: dict[int, InferenceRequest] = {}
        self._finished: dict[int, tuple[InferenceRequest, int]] = {}
        self._sent: set[int] = set()

    def initial_wakeups(self) -> tuple[ClientWakeup, ...]:
        wakeups: list[ClientWakeup] = []
        for request_id in self.plan.root_request_ids:
            turn = self.plan.turns[request_id]
            time = turn.root_arrival_time_ms
            if time is None:
                raise ValueError("根请求缺少计划到达时间")
            wakeups.append(ClientWakeup(turn.session_id, request_id, time))
        return tuple(wakeups)

    def on_wakeup(self, wakeup: ClientWakeup) -> InferenceRequest:
        turn = self.plan.turns[wakeup.turn_id]
        if turn.session_id != wakeup.session_id:
            raise ValueError("客户端唤醒的 session_id 与轮次不一致")
        if wakeup.turn_id in self._sent:
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
            if turn.root_arrival_time_ms != wakeup.send_time_ms:
                raise ValueError("根请求发送时间与计划不符")
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
        return request

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:
        request = self._inflight.get(completion.request_id)
        if request is None:
            raise ValueError("完成了未发送或已完成的请求")
        if not 1 <= completion.generated_tokens <= request.max_new_tokens:
            raise ValueError("完成的生成数超出请求可见上限")
        self._finished[request.request_id] = (
            request,
            request.prompt_len + completion.generated_tokens - 1,  # 最后一个输出尚未写入 KV
        )
        del self._inflight[completion.request_id]
        wakeups: list[ClientWakeup] = []
        for child_id in self.plan.children.get(request.request_id, ()):
            child = self.plan.turns[child_id]
            if child.think_time_ms is None:
                raise ValueError("子轮次缺少 think time")
            time = completion.finish_time_ms + child.think_time_ms  # 子轮以实际完成时间为起点
            wakeups.append(ClientWakeup(child.session_id, child_id, time))
        return tuple(wakeups)

    def on_request_aborted(self, request_id: int) -> tuple[SkippedRequest, ...]:
        """失败不触发子轮发送，递归标记全部尚未发送的后代。"""
        if request_id not in self._inflight:
            raise ValueError("失败了未发送或已结算的请求")
        del self._inflight[request_id]
        skipped: list[SkippedRequest] = []
        stack = [
            (child_id, request_id) for child_id in reversed(self.plan.children.get(request_id, ()))
        ]
        while stack:
            child_id, parent_id = stack.pop()
            if child_id in self._sent:
                raise ValueError("失败父轮的后代不应已发送")
            skipped.append(SkippedRequest(id=child_id, parent_request_id=parent_id))
            stack.extend(
                (descendant_id, child_id)
                for descendant_id in reversed(self.plan.children.get(child_id, ()))
            )
        return tuple(sorted(skipped, key=lambda item: item.id))
