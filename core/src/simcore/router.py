"""同构副本选路：根请求轮询，子请求严格继承父轮归属。"""

from simcore.schemas import InferenceRequest


class Router:
    def __init__(self, num_instances: int) -> None:  # 副本数 → 初始化根请求轮询与归属表；无返回值
        self._num_instances = num_instances
        self._next_root = 0
        self._assignments: dict[int, int] = {}

    def select(self, request: InferenceRequest) -> int:  # 请求 → 根轮询或继承父轮；返回副本 ID
        if request.parent_request_id is None:
            instance_id = self._next_root
            self._next_root = (self._next_root + 1) % self._num_instances
        else:
            instance_id = self._assignments[request.parent_request_id]  # 子轮与父轮共享缓存归属
        self._assignments[request.request_id] = instance_id
        return instance_id

    def instance_of(self, request_id: int) -> int:  # 请求 ID → 已记录的副本 ID
        return self._assignments[request_id]
