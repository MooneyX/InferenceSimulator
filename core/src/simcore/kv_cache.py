"""KV cache token 预算账本：占用记账与准入判断。

语义（见 ``docs/design/kernel.md``）：

- prefill 准入时占用 prompt 长度的额度。
- decode 每步为 batch 内每个请求 +1 token。
- 预算耗尽只禁止新请求准入；在跑请求照常增长，**不做 OOM stall**。
  因此占用可以超出预算，账本如实记录超限统计，由引擎转化为 ``warnings``。

本模块不感知时间与事件队列：它只回答「能否准入」「当前占用多少」「是否超限」，
超限发生在仿真时钟的哪一刻由调用方（引擎）记录。

依赖层级：架构上允许依赖 ``schemas``；当前实现无需其中类型，实际零依赖。
"""

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["KVCache", "KVStats"]


@dataclass(frozen=True, slots=True)
class KVStats:
    """账本统计快照，供引擎生成告警与元信息。

    Attributes:
        budget: 预算总额（token）。
        used: 当前占用。
        peak_used: 历史峰值占用。
        overflow_steps: 占用超出预算的 decode 步数。
        overflow_episodes: 超限「段」数——从未超限进入超限计一次，
            用于区分「一次持续超限」与「反复抖动」。
        forced_admissions: 以 ``force=True`` 越过预算准入的请求数。
    """

    budget: int
    used: int
    peak_used: int
    overflow_steps: int
    overflow_episodes: int
    forced_admissions: int

    @property
    def ever_overflowed(self) -> bool:
        return self.peak_used > self.budget


class KVCache:
    """token 预算账本。

    以 ``request_id -> 已占 token 数`` 记账，总占用 ``used`` 增量维护（O(1) 查询）。
    所有变更操作都校验请求存在性：重复准入、释放未知请求、推进未驻留请求
    均视为调用方编排错误而立即抛出，而非静默纠正。
    """

    __slots__ = (
        "_budget",
        "_forced_admissions",
        "_holdings",
        "_in_overflow",
        "_overflow_episodes",
        "_overflow_steps",
        "_peak_used",
        "_used",
    )

    def __init__(self, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"budget 须 >= 1，得到 {budget}")
        self._budget = budget
        self._holdings: dict[int, int] = {}
        self._used = 0
        self._peak_used = 0
        self._overflow_steps = 0
        self._overflow_episodes = 0
        self._in_overflow = False
        self._forced_admissions = 0

    # ------------------------------------------------------------ 查询

    @property
    def budget(self) -> int:
        return self._budget

    @property
    def used(self) -> int:
        return self._used

    @property
    def free(self) -> int:
        """剩余额度；超限时为负数。"""
        return self._budget - self._used

    @property
    def over_budget(self) -> bool:
        return self._used > self._budget

    def __len__(self) -> int:
        """驻留请求数。"""
        return len(self._holdings)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._holdings

    def holding(self, request_id: int) -> int:
        """某请求当前占用的 token 数。"""
        try:
            return self._holdings[request_id]
        except KeyError:
            raise KeyError(f"请求 {request_id} 未驻留于 KV cache") from None

    def can_admit(self, prompt_len: int) -> bool:
        """在当前占用下准入一个 prompt 长度为 ``prompt_len`` 的请求是否不超预算。

        只看 prompt 占用，不为其未来 decode 增长预留余量——
        增长导致的超限由「不 stall + 记告警」语义兜底。
        """
        return self._used + prompt_len <= self._budget

    def can_ever_admit(self, prompt_len: int) -> bool:
        """即便账本清空也能否容纳该 prompt。

        为 False 时正常准入永远不会成功；调度器据此决定强制准入，避免 FCFS 队头死锁。
        """
        return prompt_len <= self._budget

    def stats(self) -> KVStats:
        return KVStats(
            budget=self._budget,
            used=self._used,
            peak_used=self._peak_used,
            overflow_steps=self._overflow_steps,
            overflow_episodes=self._overflow_episodes,
            forced_admissions=self._forced_admissions,
        )

    # ------------------------------------------------------------ 变更

    def reserve(self, request_id: int, prompt_len: int, *, force: bool = False) -> None:
        """为新准入请求占用 prompt 额度。

        Args:
            request_id: 请求 ID，须未驻留。
            prompt_len: prompt token 数，须 >= 1。
            force: 允许越过预算准入（仅用于 ``can_ever_admit`` 为 False 的队头请求）。

        Raises:
            ValueError: ``prompt_len`` < 1，或请求已驻留，或非强制时预算不足。
        """
        if prompt_len < 1:
            raise ValueError(f"prompt_len 须 >= 1，得到 {prompt_len}")
        if request_id in self._holdings:
            raise ValueError(f"请求 {request_id} 重复准入")
        if not self.can_admit(prompt_len):
            if not force:
                raise ValueError(
                    f"预算不足：准入请求 {request_id} 需 {prompt_len}，剩余 {self.free}"
                )
            self._forced_admissions += 1
        self._holdings[request_id] = prompt_len
        self._used += prompt_len
        self._track_peak()

    def advance(self, request_ids: Iterable[int]) -> int:
        """一次 decode 步：每个请求 +1 token。

        在跑请求无条件增长（不 stall）。若步后占用超出预算，计入超限统计。

        Returns:
            步后超出预算的 token 数；未超限为 0。

        Raises:
            KeyError: 某请求未驻留。
        """
        holdings = self._holdings
        grown = 0
        for request_id in request_ids:
            if request_id not in holdings:
                raise KeyError(f"请求 {request_id} 未驻留于 KV cache")
            holdings[request_id] += 1
            grown += 1
        self._used += grown
        self._track_peak()

        overflow = self._used - self._budget
        if overflow > 0:
            self._overflow_steps += 1
            if not self._in_overflow:
                self._overflow_episodes += 1
                self._in_overflow = True
            return overflow
        self._in_overflow = False
        return 0

    def release(self, request_id: int) -> int:
        """请求完成，归还其全部占用。

        Returns:
            归还的 token 数。

        Raises:
            KeyError: 请求未驻留。
        """
        try:
            tokens = self._holdings.pop(request_id)
        except KeyError:
            raise KeyError(f"请求 {request_id} 未驻留于 KV cache") from None
        self._used -= tokens
        if not self.over_budget:
            self._in_overflow = False
        return tokens

    def _track_peak(self) -> None:
        if self._used > self._peak_used:
            self._peak_used = self._used
