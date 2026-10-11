"""计划化 trace 的开环时钟与按父完成驱动的闭环客户端。"""

import math
import random
from collections import Counter

from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    InferenceRequest,
    RequestCompletion,
    SkippedRequest,
    TracePlan,
    TurnPlan,
    WorkloadPlan,
)
from simcore.workload import _visible_limit


class TraceClient:
    def __init__(
        self, source: TracePlan, engine: EngineConfig, *, mode: str, session_rate: float, seed: int
    ) -> None:
        if (
            mode not in ("open_loop", "session")
            or not math.isfinite(session_rate)
            or session_rate <= 0
            or seed < 0
        ):
            raise ValueError("无效的 trace 投放模式、会话速率或随机种子")
        if source.block_size != engine.page_size:
            raise ValueError("trace 块大小必须与引擎 page_size 一致")
        self.source = source
        self.mode = mode
        self._sent: set[int] = set()
        self._completed: set[int] = set()
        self._children: dict[int, list[int]] = {}
        self._turns = {item.request_id: item for item in source.requests}
        seen = Counter(
            (item.session_id, block) for item in source.requests for block in item.hash_ids
        )
        rng = random.Random(seed)
        starts: list[float] = []
        clock = 0.0
        for _ in source.sessions:
            clock += rng.expovariate(session_rate / 1000)
            starts.append(clock)
        self._wakeups = tuple(
            ClientWakeup(
                item.session_id, item.request_id, starts[item.session_id] + item.arrival_offset_ms
            )
            for item in source.requests
            if mode == "open_loop" or item.parent_request_id is None
        )
        turns: dict[int, TurnPlan] = {}
        for item in source.requests:
            if item.parent_request_id is not None:
                self._children.setdefault(item.parent_request_id, []).append(item.request_id)
            reusable = 0
            for block in item.hash_ids[: (item.prompt_len - 1) // engine.page_size]:
                if seen[item.session_id, block] < 2:
                    break
                reusable += 1
            turns[item.request_id] = TurnPlan(
                request_id=item.request_id,
                session_id=item.session_id,
                parent_request_id=item.parent_request_id,
                root_arrival_time_ms=(
                    starts[item.session_id] + item.arrival_offset_ms
                    if mode == "open_loop" or item.parent_request_id is None
                    else None
                ),
                think_time_ms=item.think_time_ms,
                prompt_len=item.prompt_len,
                output_len=item.output_len,
                max_new_tokens=_visible_limit(item.prompt_len, max(1, item.output_len), engine),
                ideal_reuse_blocks=reusable,
            )
        first_by_session: dict[int, int] = {}
        for item in source.requests:
            first_by_session.setdefault(item.session_id, item.request_id)
        total = sum(item.prompt_len for item in source.requests)
        self.plan = WorkloadPlan(
            turns=turns,
            root_request_ids=tuple(first_by_session.values()),
            children={key: tuple(values) for key, values in self._children.items()},
            constructed_ideal_prefix_hit_rate=(
                sum(turn.ideal_reuse_blocks for turn in turns.values()) * engine.page_size / total
            ),
        )

    def initial_wakeups(self) -> tuple[ClientWakeup, ...]:
        return self._wakeups

    def on_wakeup(self, wakeup: ClientWakeup) -> InferenceRequest:
        turn = self.plan.turns[wakeup.turn_id]
        if wakeup.turn_id in self._sent or turn.session_id != wakeup.session_id:
            raise ValueError("重复或无效的 trace 唤醒")
        if (
            self.mode == "session"
            and turn.parent_request_id is not None
            and turn.parent_request_id not in self._completed
        ):
            raise ValueError("依赖的父请求尚未完成")
        self._sent.add(wakeup.turn_id)
        source = self._turns[wakeup.turn_id]
        return InferenceRequest(
            request_id=turn.request_id,
            session_id=turn.session_id,
            parent_request_id=turn.parent_request_id,
            prompt_len=turn.prompt_len,
            max_new_tokens=turn.max_new_tokens,
            inherited_prefix_ids=(),
            ideal_reuse_blocks=turn.ideal_reuse_blocks,
            trace_hash_ids=source.hash_ids,
            is_trace=True,
        )

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:
        self._completed.add(completion.request_id)
        if self.mode == "open_loop":
            return ()
        return tuple(
            ClientWakeup(
                self._turns[child].session_id,
                child,
                completion.finish_time_ms + (self._turns[child].think_time_ms or 0),
            )
            for child in self._children.get(completion.request_id, ())
        )

    def on_request_aborted(self, request_id: int) -> tuple[SkippedRequest, ...]:
        if self.mode == "open_loop":
            return ()
        skipped: list[SkippedRequest] = []
        stack = [(child, request_id) for child in self._children.get(request_id, ())]
        while stack:
            child, parent = stack.pop()
            skipped.append(SkippedRequest(id=child, parent_request_id=parent))
            stack.extend((grandchild, child) for grandchild in self._children.get(child, ()))
        return tuple(sorted(skipped, key=lambda item: item.id))
