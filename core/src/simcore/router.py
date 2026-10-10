"""同构副本选路：根请求轮询，子请求严格继承父轮归属。"""

from simcore.schemas import InferenceRequest


class Router:
    def __init__(self, num_instances: int) -> None:
        self._num_instances = num_instances
        self._next_root = 0
        self._assignments: dict[int, int] = {}

    def select(self, request: InferenceRequest) -> int:
        if request.parent_request_id is None:
            instance_id = self._next_root
            self._next_root = (self._next_root + 1) % self._num_instances
        else:
            instance_id = self._assignments[request.parent_request_id]
        self._assignments[request.request_id] = instance_id
        return instance_id

    def instance_of(self, request_id: int) -> int:
        return self._assignments[request_id]
