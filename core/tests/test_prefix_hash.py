"""块身份整数编码与 Trace 前缀树重编号。"""

from itertools import pairwise

from simcore.prefix_hash import CHUNK_OFFSET_BITS, chunk_base, prefix_tree_order, stable_derive


def runs(ids: list[int]) -> int:
    return 1 + sum(following != previous + 1 for previous, following in pairwise(ids))


def test_chunk_base_separates_owner_domains() -> None:
    assert chunk_base(0) == 0
    assert chunk_base(1) == 1 << CHUNK_OFFSET_BITS
    assert chunk_base(3) + (1 << CHUNK_OFFSET_BITS) - 1 < chunk_base(4)


def test_prefix_tree_order_numbers_heavy_branch_contiguously() -> None:
    # 共享前缀 0-1 后分叉：主线 2-3-4 被三次输入经过，支线 9 仅一次。
    sequences = [[0, 1], [0, 1, 9], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    order = prefix_tree_order(sequences)
    assert order == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 9: 5}
    assert [runs([order[block] for block in blocks]) for blocks in sequences] == [1, 2, 1, 1, 1]


def test_prefix_tree_order_handles_multiple_roots_and_empty_inputs() -> None:
    order = prefix_tree_order([[5, 6], [], [7], [5, 6, 8], [7, 1]])
    assert sorted(order.values()) == list(range(5))
    assert order[6] == order[5] + 1 and order[8] == order[6] + 1
    assert order[1] == order[7] + 1


def test_stable_derive_remains_domain_specific() -> None:
    assert stable_derive(17, "length") == stable_derive(17, "length")
    assert stable_derive(17, "length") != stable_derive(17, "root-arrival")
