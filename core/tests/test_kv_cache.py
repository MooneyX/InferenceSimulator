"""kv_cache 模块单测：记账守恒、准入判断、不 stall 语义、超限统计与编排错误防御。"""

import pytest

from simcore.kv_cache import KVCache, KVStats


class TestConstruction:
    def test_initial_state(self) -> None:
        kv = KVCache(100)
        assert (kv.budget, kv.used, kv.free, len(kv)) == (100, 0, 100, 0)
        assert not kv.over_budget

    @pytest.mark.parametrize("budget", [0, -1])
    def test_rejects_non_positive_budget(self, budget: int) -> None:
        with pytest.raises(ValueError, match="budget"):
            KVCache(budget)


class TestAdmission:
    def test_can_admit_boundary_inclusive(self) -> None:
        """恰好用满预算允许准入。"""
        kv = KVCache(100)
        kv.reserve(0, 60)
        assert kv.can_admit(40)
        assert not kv.can_admit(41)

    def test_reserve_occupies_prompt_len(self) -> None:
        kv = KVCache(100)
        kv.reserve(7, 30)
        assert (kv.used, kv.holding(7), len(kv)) == (30, 30, 1)
        assert 7 in kv

    def test_reserve_over_budget_rejected_without_force(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 90)
        with pytest.raises(ValueError, match="预算不足"):
            kv.reserve(1, 20)
        assert kv.used == 90 and 1 not in kv

    def test_duplicate_reserve_rejected(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        with pytest.raises(ValueError, match="重复准入"):
            kv.reserve(0, 10)

    @pytest.mark.parametrize("prompt_len", [0, -3])
    def test_reserve_rejects_non_positive(self, prompt_len: int) -> None:
        with pytest.raises(ValueError, match="prompt_len"):
            KVCache(100).reserve(0, prompt_len)

    def test_can_ever_admit(self) -> None:
        kv = KVCache(100)
        assert kv.can_ever_admit(100)
        assert not kv.can_ever_admit(101)

    def test_forced_admission_counts_only_when_actually_over(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 50, force=True)  # 预算内，force 无实际作用
        kv.reserve(1, 200, force=True)  # 真正越过预算
        stats = kv.stats()
        assert stats.forced_admissions == 1
        assert kv.used == 250 and kv.over_budget


class TestDecodeGrowth:
    def test_advance_adds_one_per_request(self) -> None:
        kv = KVCache(1000)
        kv.reserve(0, 10)
        kv.reserve(1, 20)
        assert kv.advance([0, 1]) == 0
        assert (kv.holding(0), kv.holding(1), kv.used) == (11, 21, 32)

    def test_advance_subset(self) -> None:
        kv = KVCache(1000)
        kv.reserve(0, 10)
        kv.reserve(1, 20)
        kv.advance([1])
        assert (kv.holding(0), kv.holding(1)) == (10, 21)

    def test_advance_empty_is_noop(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        assert kv.advance([]) == 0
        assert kv.used == 10

    def test_advance_accepts_iterator(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        kv.reserve(1, 10)
        kv.advance(iter([0, 1]))
        assert kv.used == 22

    def test_advance_unknown_request_rejected(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        with pytest.raises(KeyError, match="未驻留"):
            kv.advance([0, 99])

    def test_no_stall_growth_continues_past_budget(self) -> None:
        """核心语义：预算耗尽不阻止在跑请求增长，返回超出量。"""
        kv = KVCache(10)
        kv.reserve(0, 5)
        kv.reserve(1, 5)
        assert kv.advance([0, 1]) == 2
        assert kv.advance([0, 1]) == 4
        assert kv.used == 14 and kv.free == -4 and kv.over_budget

    def test_admission_blocked_while_over_budget(self) -> None:
        kv = KVCache(10)
        kv.reserve(0, 10)
        kv.advance([0])
        assert not kv.can_admit(1)


class TestRelease:
    def test_release_returns_all_tokens(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        kv.advance([0])
        kv.advance([0])
        assert kv.release(0) == 12
        assert (kv.used, len(kv)) == (0, 0)
        assert 0 not in kv

    def test_release_unknown_rejected(self) -> None:
        with pytest.raises(KeyError, match="未驻留"):
            KVCache(100).release(0)

    def test_double_release_rejected(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 10)
        kv.release(0)
        with pytest.raises(KeyError):
            kv.release(0)

    def test_holding_unknown_rejected(self) -> None:
        with pytest.raises(KeyError, match="未驻留"):
            KVCache(100).holding(3)

    def test_release_restores_admission(self) -> None:
        kv = KVCache(10)
        kv.reserve(0, 10)
        assert not kv.can_admit(5)
        kv.release(0)
        assert kv.can_admit(10)


class TestConservation:
    def test_used_equals_sum_of_holdings(self) -> None:
        """增量维护的 used 必须始终等于逐请求占用之和。"""
        kv = KVCache(10_000)
        live: set[int] = set()
        for step in range(200):
            if step % 3 == 0:
                kv.reserve(step, 1 + step % 17)
                live.add(step)
            kv.advance(sorted(live))
            if step % 7 == 6 and live:
                kv.release(min(live))
                live.remove(min(live))
            assert kv.used == sum(kv.holding(r) for r in live)
        for r in sorted(live):
            kv.release(r)
        assert kv.used == 0


class TestStats:
    def test_peak_tracks_maximum(self) -> None:
        kv = KVCache(100)
        kv.reserve(0, 40)
        kv.reserve(1, 30)
        kv.release(0)
        kv.advance([1])
        stats = kv.stats()
        assert (stats.used, stats.peak_used) == (31, 70)
        assert not stats.ever_overflowed

    def test_overflow_steps_and_episodes(self) -> None:
        """连续超限只算一段；回落后再超限算新的一段。"""
        kv = KVCache(10)
        kv.reserve(0, 9)
        kv.advance([0])  # 10：未超
        kv.advance([0])  # 11：进入第 1 段
        kv.advance([0])  # 12：同一段
        kv.release(0)  # 0：回落
        kv.reserve(1, 10)
        kv.advance([1])  # 11：第 2 段
        stats = kv.stats()
        assert stats.overflow_steps == 3
        assert stats.overflow_episodes == 2
        assert stats.peak_used == 12
        assert stats.ever_overflowed

    def test_episode_ends_when_advance_returns_within_budget(self) -> None:
        kv = KVCache(10)
        kv.reserve(0, 10)
        kv.reserve(1, 1, force=True)
        kv.advance([0])  # 12：第 1 段
        kv.release(1)  # 11：仍超限，段未结束
        kv.advance([0])  # 12：同一段
        assert kv.stats().overflow_episodes == 1

    def test_stats_is_immutable_snapshot(self) -> None:
        kv = KVCache(100)
        snapshot = kv.stats()
        kv.reserve(0, 50)
        assert snapshot.used == 0
        assert isinstance(snapshot, KVStats)
        with pytest.raises(AttributeError):
            snapshot.used = 1  # type: ignore[misc]
