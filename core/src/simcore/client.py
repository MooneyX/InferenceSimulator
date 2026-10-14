"""统一客户端：合成/Trace 请求构造与开环/闭环投放。"""

import heapq
import math
import random
from collections import Counter
from typing import Literal, cast

from simcore.prefix_hash import logical_chunk_id, stable_derive
from simcore.schemas import (
    ClientWakeup,
    EngineConfig,
    InferenceRequest,
    RequestCompletion,
    SkippedRequest,
    SyntheticWorkloadConfig,
    TracePlan,
    TurnPlan,
    WorkloadConfig,
    WorkloadMode,
    WorkloadPlan,
)
from simcore.workload import (
    build_synthetic_closed_loop_plan,
    request_output_limit,
    sample_length,
)

__all__ = ["Client"]


class Client:
    """持有请求生命周期；只在发送时构造请求，不参与缓存或调度决策。"""

    def __init__(
        self,
        workload: WorkloadConfig,
        engine: EngineConfig,
        *,
        trace_plan: TracePlan | None = None,
    ) -> None:
        self.engine = engine
        self._sent: set[int] = set()
        self._inflight: dict[int, InferenceRequest] = {}
        self._workload_mode: WorkloadMode
        self._prepare_workload(workload, trace_plan=trace_plan)

    def _prepare_workload(
        self, workload: WorkloadConfig, *, trace_plan: TracePlan | None = None
    ) -> None:
        """统一准备四种负载的计划或动态生成状态；请求仍在发送时构造。"""
        self._workload_cfg = workload
        self._workload_mode = workload.mode
        self._arrival_mode = workload.arrival_mode
        if workload.mode == "trace":
            if trace_plan is None:
                raise ValueError("Trace 仿真须提供 trace_plan 请求计划")
            if self.arrival_mode == "closed_loop":
                self._completed: set[int] = set()
            self._prepare_trace(trace_plan, workload)
        else:
            if trace_plan is not None:
                raise ValueError("合成仿真不能提供 trace_plan 请求计划")
            self._synthetic_cfg = cast(SyntheticWorkloadConfig, workload.synthetic)
            self._finished: dict[int, tuple[InferenceRequest, int]] = {}
            if self.arrival_mode == "closed_loop":
                self._plan = build_synthetic_closed_loop_plan(workload, self.engine)
            else:
                self._roots_rng = random.Random(stable_derive(workload.seed, "root-length"))
                self._increments_rng = random.Random(
                    stable_derive(workload.seed, "turn-increment")
                )
                self._outputs_rng = random.Random(
                    stable_derive(workload.seed, "output-length")
                )
                self._ready: list[tuple[float, int, int]] = []
                self._turns: dict[int, TurnPlan] = {}
                self._roots: list[int] = []
                self._children: dict[int, tuple[int, ...]] = {}

    @property
    def workload_mode(self) -> WorkloadMode:
        return self._workload_mode

    @property
    def arrival_mode(self) -> Literal["closed_loop", "open_loop"]:
        return self._arrival_mode

    @property
    def _dynamic_requests(self) -> bool:
        return self.workload_mode == "synthetic" and self.arrival_mode == "open_loop"

    @property
    def plan(self) -> WorkloadPlan:
        if not self._dynamic_requests:
            return self._plan
        total = sum(turn.prompt_len for turn in self._turns.values())
        ideal = sum(turn.ideal_reuse_blocks for turn in self._turns.values())
        return WorkloadPlan(
            turns=self._turns,
            root_request_ids=tuple(self._roots),
            children=self._children,
            constructed_ideal_prefix_hit_rate=(
                ideal * self.engine.page_size / total if total else 0.0
            ),
        )

    def _prepare_trace(
        self, source: TracePlan, workload: WorkloadConfig
    ) -> None:
        if source.block_size != self.engine.page_size:
            raise ValueError("trace 块大小必须与引擎 page_size 一致")
        self._trace_turns = {item.request_id: item for item in source.requests}
        seen = Counter(
            (item.session_id, block) for item in source.requests for block in item.hash_ids
        )
        rng = random.Random(workload.seed)
        starts: list[float] = []
        clock = 0.0
        for _ in source.sessions:
            clock += rng.expovariate(cast(float, workload.session_rate) / 1000)
            starts.append(clock)
        arrivals = [starts[item.session_id] + item.arrival_offset_ms for item in source.requests]
        if workload.target_qps is not None:
            if len(arrivals) < 2 or max(arrivals) == min(arrivals):
                raise ValueError("目标 QPS 至少需要两个到达时刻不同的请求")
            first = min(arrivals)
            scale = (len(arrivals) - 1) * 1000 / (workload.target_qps * (max(arrivals) - first))
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("目标 QPS 超出当前时间精度可表示的范围")
            arrivals = [first + (arrival - first) * scale for arrival in arrivals]
            if not all(math.isfinite(arrival) for arrival in arrivals) or max(arrivals) == min(
                arrivals
            ):
                raise ValueError("缩放后的请求时刻超出可表示范围")
        self._trace_wakeups = tuple(
            ClientWakeup(item.session_id, item.request_id, arrivals[index])
            for index, item in enumerate(source.requests)
            if self.arrival_mode == "open_loop" or item.parent_request_id is None
        )
        turns: dict[int, TurnPlan] = {}
        children: dict[int, list[int]] = {}
        first_by_session: dict[int, int] = {}
        for index, item in enumerate(source.requests):
            first_by_session.setdefault(item.session_id, item.request_id)
            if item.parent_request_id is not None:
                children.setdefault(item.parent_request_id, []).append(item.request_id)
            reusable = 0
            for block in item.hash_ids[: (item.prompt_len - 1) // self.engine.page_size]:
                if seen[item.session_id, block] < 2:
                    break
                reusable += 1
            turns[item.request_id] = TurnPlan(
                request_id=item.request_id,
                session_id=item.session_id,
                parent_request_id=item.parent_request_id,
                root_arrival_time_ms=(
                    arrivals[index]
                    if self.arrival_mode == "open_loop" or item.parent_request_id is None
                    else None
                ),
                think_time_ms=item.think_time_ms,
                prompt_len=item.prompt_len,
                output_len=item.output_len,
                max_new_tokens=request_output_limit(item.output_len, self.engine.max_output_tokens),
                ideal_reuse_blocks=reusable,
            )
        total = sum(item.prompt_len for item in source.requests)
        self._plan = WorkloadPlan(
            turns=turns,
            root_request_ids=tuple(first_by_session.values()),
            children={key: tuple(values) for key, values in children.items()},
            constructed_ideal_prefix_hit_rate=(
                sum(turn.ideal_reuse_blocks for turn in turns.values())
                * self.engine.page_size
                / total
            ),
        )

    def initial_wakeups(self) -> tuple[ClientWakeup, ...]:
        if self.workload_mode == "trace":
            return self._trace_wakeups
        if self.arrival_mode == "open_loop":
            rng = random.Random(stable_derive(self._workload_cfg.seed, "open-loop-arrival"))
            rate_per_ms = cast(float, self._workload_cfg.target_qps) / 1000.0
            clock = 0.0
            wakeups = []
            for index in range(self._synthetic_cfg.num_requests):
                clock += rng.expovariate(rate_per_ms)
                wakeups.append(ClientWakeup(index, index, clock))
            return tuple(wakeups)
        wakeups = []
        for request_id in self._plan.root_request_ids:
            turn = self._plan.turns[request_id]
            time = turn.root_arrival_time_ms
            if time is None:
                raise ValueError("根请求缺少计划到达时间")
            wakeups.append(ClientWakeup(turn.session_id, request_id, time))
        return tuple(wakeups)

    def _next_synthetic_turn(self, wakeup: ClientWakeup) -> TurnPlan:
        key = wakeup.turn_id
        if self._ready:
            _, session_id, parent_id = heapq.heappop(self._ready)
            parent = self._turns[parent_id]
            increment = sample_length(
                self._synthetic_cfg.turn_increment_len, self._increments_rng
            )
            prompt_len = parent.prompt_len + parent.output_len + increment
            inherited = min(
                (parent.prompt_len + parent.output_len - 1) // self.engine.page_size,
                (prompt_len - 1) // self.engine.page_size,
            )
            self._children[parent_id] = (key,)
        else:
            session_id = key
            parent_id = None
            prompt_len = sample_length(self._synthetic_cfg.prompt_len, self._roots_rng)
            inherited = 0
            self._roots.append(key)
        output = sample_length(self._synthetic_cfg.output_len, self._outputs_rng)
        maximum = request_output_limit(output, self.engine.max_output_tokens)
        turn = TurnPlan(
            key,
            session_id,
            parent_id,
            wakeup.send_time_ms if parent_id is None else None,
            None,
            prompt_len,
            output,
            maximum,
            inherited,
        )
        self._turns[key] = turn
        return turn

    def on_wakeup(self, wakeup: ClientWakeup) -> tuple[InferenceRequest, int]:
        if wakeup.turn_id in self._sent:
            raise ValueError("轮次未就绪、重复发送或发送时间不符")
        dynamic = self._dynamic_requests
        turn = self._next_synthetic_turn(wakeup) if dynamic else self._plan.turns[wakeup.turn_id]
        if not dynamic and turn.session_id != wakeup.session_id:
            raise ValueError("客户端唤醒的 session_id 与轮次不一致")
        head: tuple[str, ...] = ()
        hashes: tuple[int, ...] = ()
        if self.workload_mode == "trace":
            if (
                self.arrival_mode == "closed_loop"
                and turn.parent_request_id is not None
                and turn.parent_request_id not in self._completed
            ):
                raise ValueError("依赖的父请求尚未完成")
            hashes = self._trace_turns[turn.request_id].hash_ids
        else:
            inherited = turn.ideal_reuse_blocks
            if inherited > (turn.prompt_len - 1) // self.engine.page_size:
                raise ValueError("继承页超过子轮输入的完整前缀")
            if turn.parent_request_id is not None:
                parent = self._finished.get(turn.parent_request_id)
                if parent is None or parent[1] // self.engine.page_size < inherited:
                    raise ValueError("子轮次需要父请求完成后的完整块上下文")
                head = tuple(
                    logical_chunk_id(parent[0].request_id, index, parent[0].inherited_prefix_ids)
                    for index in range(inherited)
                )
                if dynamic:
                    del self._finished[turn.parent_request_id]
            else:
                if turn.root_arrival_time_ms != wakeup.send_time_ms:
                    raise ValueError("根请求发送时间与计划不符")
                if inherited:
                    raise ValueError("根请求不能继承完整块")
        request = InferenceRequest(
            request_id=turn.request_id,
            session_id=turn.session_id,
            parent_request_id=turn.parent_request_id,
            prompt_len=turn.prompt_len,
            max_new_tokens=turn.max_new_tokens,
            inherited_prefix_ids=head,
            ideal_reuse_blocks=turn.ideal_reuse_blocks,
            trace_hash_ids=hashes,
            is_trace=self.workload_mode == "trace",
        )
        self._sent.add(turn.request_id)
        self._inflight[turn.request_id] = request
        return request, turn.output_len

    def on_request_completed(self, completion: RequestCompletion) -> tuple[ClientWakeup, ...]:
        request = self._inflight.get(completion.request_id)
        if request is None:
            raise ValueError("完成了未发送或已完成的请求")
        if not 1 <= completion.generated_tokens <= request.max_new_tokens:
            raise ValueError("完成的生成数超出请求可见上限")
        del self._inflight[completion.request_id]
        if self.workload_mode == "synthetic":
            self._finished[request.request_id] = (
                request,
                request.prompt_len + completion.generated_tokens - 1,
            )
            if self.arrival_mode == "open_loop":
                heapq.heappush(
                    self._ready, (completion.finish_time_ms, request.session_id, request.request_id)
                )
        elif self.arrival_mode == "closed_loop":
            self._completed.add(request.request_id)
        if self.arrival_mode == "open_loop":
            return ()
        wakeups = []
        for child_id in self._plan.children.get(request.request_id, ()):
            child = self._plan.turns[child_id]
            if child.think_time_ms is None and self.workload_mode == "synthetic":
                raise ValueError("子轮次缺少 think time")
            time = completion.finish_time_ms + (child.think_time_ms or 0)
            wakeups.append(ClientWakeup(child.session_id, child_id, time))
        return tuple(wakeups)

    def on_request_aborted(self, request_id: int) -> tuple[SkippedRequest, ...]:
        if request_id not in self._inflight:
            raise ValueError("失败了未发送或已结算的请求")
        del self._inflight[request_id]
        if self.arrival_mode == "open_loop":
            return ()
        skipped = []
        stack = [(child, request_id) for child in reversed(self._plan.children.get(request_id, ()))]
        while stack:
            child_id, parent_id = stack.pop()
            if child_id in self._sent:
                raise ValueError("失败父轮的后代不应已发送")
            skipped.append(SkippedRequest(id=child_id, parent_request_id=parent_id))
            stack.extend(
                (descendant, child_id)
                for descendant in reversed(self._plan.children.get(child_id, ()))
            )
        return tuple(sorted(skipped, key=lambda item: item.id))
