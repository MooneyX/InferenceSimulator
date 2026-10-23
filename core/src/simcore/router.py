"""同构副本选路：带亲和的轮询，或逐请求最低 token 负载优先。"""

from collections.abc import Sequence
from typing import Literal

from simcore.schemas import InferenceRequest


class Router:
    def __init__(
        self,
        num_instances: int,
        *,
        policy: Literal["round_robin", "least_load"] = "round_robin",
        session_affinity: bool = False,
    ) -> None:
        self.policy: Literal["round_robin", "least_load"] = policy
        self._num_instances = num_instances
        self._next_instance: int = 0
        self._assignments: dict[int, int] = {}
        self._sessions: dict[int, int] = {}
        self._session_affinity = session_affinity

    def select(self, request: InferenceRequest, loads: Sequence[int] = ()) -> int:
        """按亲和轮询或当前 token 负载选路；并列最低者从游标起轮询。"""
        if self.policy == "least_load":
            minimum = min(loads)
            instance_id = next(
                index
                for offset in range(self._num_instances)
                if loads[index := (self._next_instance + offset) % self._num_instances] == minimum
            )
            self._next_instance = (instance_id + 1) % self._num_instances
        elif self._session_affinity and request.session_id in self._sessions:
            instance_id = self._sessions[request.session_id]
        elif request.parent_request_id is not None and not self._session_affinity:
            instance_id = self._assignments[request.parent_request_id]
        else:
            instance_id = self._next_instance
            self._next_instance = (self._next_instance + 1) % self._num_instances
        self._sessions[request.session_id] = instance_id
        self._assignments[request.request_id] = instance_id
        return instance_id

    def instance_of(self, request_id: int) -> int:  # 请求 ID → 已记录的副本 ID
        return self._assignments[request_id]
