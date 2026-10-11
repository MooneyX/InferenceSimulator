"""同构副本选路：根请求轮询，子请求严格继承父轮归属。"""

from simcore.schemas import InferenceRequest


class Router:
    def __init__(self, num_instances: int, *, session_affinity: bool = False) -> None:
        self._num_instances = num_instances
        self._next_root = 0
        self._assignments: dict[int, int] = {}
        self._sessions: dict[int, int] = {}
        self._session_affinity = session_affinity

    def select(self, request: InferenceRequest) -> int:  # 请求 → 会话亲和或父子亲和；返回副本 ID
        if self._session_affinity and request.session_id in self._sessions:
            instance_id = self._sessions[request.session_id]
        elif request.parent_request_id is not None and not self._session_affinity:
            instance_id = self._assignments[request.parent_request_id]
        else:
            instance_id = self._next_root
            self._next_root = (self._next_root + 1) % self._num_instances
        self._sessions[request.session_id] = instance_id
        self._assignments[request.request_id] = instance_id
        return instance_id

    def instance_of(self, request_id: int) -> int:  # 请求 ID → 已记录的副本 ID
        return self._assignments[request_id]
