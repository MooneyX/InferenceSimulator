"""会话客户端：闭环回调或独立开环时钟驱动下一轮发送。"""

import heapq
import random
from typing import cast

from simcore.prefix_hash import logical_chunk_id, stable_derive
from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    InferenceRequest,
    LengthDist,
    RequestCompletion,
    SkippedRequest,
    SyntheticWorkloadConfig,
    TurnPlan,
    WorkloadPlan,
)
from simcore.workload import _visible_limit, sample_length

__all__ = ["ClientDriver", "OpenLoopClient"]


class ClientDriver:
    """根据已完成父轮的请求身份和实际生成数构造继承前缀。"""

    def __init__(self, plan: WorkloadPlan, engine: EngineConfig) -> None:  # 轮次计划与引擎配置 → 初始化闭环客户端；无返回值
        self.plan = plan
        self.page_size = engine.page_size
        self._inflight: dict[int, InferenceRequest] = {}
        self._finished: dict[int, tuple[InferenceRequest, int]] = {}
        self._sent: set[int] = set()

    def initial_wakeups(self) -> tuple[ClientWakeup, ...]:  # 计划中的根轮次 → 初始唤醒事件元组
        wakeups: list[ClientWakeup] = []
        for request_id in self.plan.root_request_ids:
            turn = self.plan.turns[request_id]
            time = turn.root_arrival_time_ms
            if time is None:
                raise ValueError("根请求缺少计划到达时间")
            wakeups.append(ClientWakeup(turn.session_id, request_id, time))
        return tuple(wakeups)

    def on_wakeup(self, wakeup: ClientWakeup) -> InferenceRequest:  # 轮次唤醒消息 → 携带父轮继承页的推理请求
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

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:  # 父轮完成消息 → 各子轮定时唤醒事件
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

    def on_request_aborted(self, request_id: int) -> tuple[SkippedRequest, ...]:  # 失败的请求 ID → 全部未发送后代的跳过记录
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


class OpenLoopClient:
    """每次时钟投放一个请求：先取已完成会话，否则创建新的根会话。"""

    def __init__(self, config: SyntheticWorkloadConfig, engine: EngineConfig) -> None:  # 开环负载与引擎配置 → 初始化采样流和待续会话；无返回值
        self.config = config
        self.engine = engine
        self._increment_dist = cast(LengthDist, config.turn_increment_len)
        self._roots_rng = random.Random(stable_derive(config.seed, "root-length"))
        self._increments_rng = random.Random(stable_derive(config.seed, "turn-increment"))
        self._outputs_rng = random.Random(stable_derive(config.seed, "output-length"))
        self._ready: list[tuple[float, int, int]] = []
        self._inflight: dict[int, InferenceRequest] = {}
        self._turns: dict[int, TurnPlan] = {}
        self._roots: list[int] = []
        self._children: dict[int, tuple[int, ...]] = {}
        self._next_id = 0

    def send(self, now_ms: float) -> tuple[InferenceRequest, int]:  # 当前开环投放时刻 → 新请求及其实际输出长度
        key = self._next_id
        self._next_id += 1
        if self._ready:
            _, session_id, parent_id = heapq.heappop(self._ready)
            parent = self._inflight.pop(parent_id)
            parent_turn = self._turns[parent_id]
            increment = sample_length(self._increment_dist, self._increments_rng)
            prompt_len = parent_turn.prompt_len + parent_turn.output_len + increment
            inherited = min(
                (parent_turn.prompt_len + parent_turn.output_len - 1) // self.engine.page_size,
                (prompt_len - 1) // self.engine.page_size,
            )
            head = tuple(
                logical_chunk_id(parent_id, index, parent.inherited_prefix_ids)
                for index in range(inherited)
            )
            self._children[parent_id] = (key,)
        else:
            session_id = key
            parent_id = None
            prompt_len = sample_length(self.config.prompt_len, self._roots_rng)
            inherited = 0
            head = ()
            self._roots.append(key)
        max_new_tokens = _visible_limit(prompt_len, self.config.max_new_tokens, self.engine)
        output_len = min(sample_length(self.config.output_len, self._outputs_rng), max_new_tokens)
        self._turns[key] = TurnPlan(
            request_id=key,
            session_id=session_id,
            parent_request_id=parent_id,
            root_arrival_time_ms=now_ms if parent_id is None else None,
            think_time_ms=None,
            prompt_len=prompt_len,
            output_len=output_len,
            max_new_tokens=max_new_tokens,
            ideal_reuse_blocks=inherited,
        )
        request = InferenceRequest(
            request_id=key,
            session_id=session_id,
            parent_request_id=parent_id,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            inherited_prefix_ids=head,
            ideal_reuse_blocks=inherited,
        )
        self._inflight[key] = request
        return request, output_len

    def on_request_completed(self, completion: RequestCompletion) -> None:  # 完成消息 → 将会话放入待续堆；无返回值
        # 完成只使会话就绪；下一轮仍须等独立投放时钟。
        heapq.heappush(
            self._ready,
            (completion.finish_time_ms, self._inflight[completion.request_id].session_id,
             completion.request_id),
        )

    def on_request_aborted(self, request_id: int) -> None:  # 失败请求 ID → 移出飞行中集合；无返回值
        self._inflight.pop(request_id)

    @property
    def plan(self) -> WorkloadPlan:  # 已投放的轮次及会话 → 当前开环负载计划
        total = sum(turn.prompt_len for turn in self._turns.values())
        ideal = sum(turn.ideal_reuse_blocks for turn in self._turns.values())
        return WorkloadPlan(
            turns=self._turns,
            root_request_ids=tuple(self._roots),
            children=self._children,
            constructed_ideal_prefix_hit_rate=ideal * self.engine.page_size / total,
        )
