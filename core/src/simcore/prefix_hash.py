"""KV 前缀块身份与随机流种子派生；仅完整块进入共享目录。"""

import hashlib
import json
from collections.abc import Sequence

CHUNK_OFFSET_BITS = 32  # 块 ID = 归属域 << 32 | 域内偏移


def chunk_base(owner: int) -> int:  # 归属域编号 → 该域首个块 ID
    """合成会话、Trace 会话与 Trace 请求尾页各占一个归属域，域内偏移连续。"""
    return owner << CHUNK_OFFSET_BITS


def prefix_tree_order(sequences: Sequence[Sequence[int]]) -> dict[int, int]:
    """同一会话的块序列 → 前缀树重子优先先序编号，使多数输入的块 ID 连续。

    块 ID 已按 (位置, 前驱) 唯一，会话内所有输入构成一棵前缀树；某块首次出现后其后续块
    必然也是首次出现，故二分定位新块起点即可。重子按经过该节点的输入数选择。
    """
    parent: dict[int, int | None] = {}
    ends: dict[int, int] = {}
    for blocks in sequences:
        if not blocks:
            continue
        low, high = 0, len(blocks)
        while low < high:
            middle = (low + high) // 2
            if blocks[middle] in parent:
                low = middle + 1
            else:
                high = middle
        previous = blocks[low - 1] if low else None
        for block in blocks[low:]:
            parent[block] = previous
            previous = block
        ends[blocks[-1]] = ends.get(blocks[-1], 0) + 1
    weight = dict.fromkeys(parent, 0)
    children: dict[int | None, list[int]] = {}
    for block, count in ends.items():
        weight[block] = count
    for block in reversed(parent):  # 子节点总在父节点之后创建
        above = parent[block]
        if above is not None:
            weight[above] += weight[block]
    for block, above in parent.items():
        children.setdefault(above, []).append(block)
    order: dict[int, int] = {}
    stack = list(reversed(children.get(None, ())))
    while stack:
        block = stack.pop()
        order[block] = len(order)
        below = children.get(block)
        if below:
            heavy = max(below, key=weight.__getitem__)
            stack.extend(item for item in reversed(below) if item != heavy)
            stack.append(heavy)
    return order


def stable_derive(seed: int, domain: str) -> int:  # 随机种子与用途域 → 稳定独立的 64 位种子
    """按域派生独立随机流，避免采样维度之间相互扰动。"""
    payload = json.dumps([seed, domain], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
