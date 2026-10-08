"""KV 前缀块身份与随机流种子派生；仅完整块进入共享目录。"""

import hashlib
import json


def encode(request_id: int, block_index: int) -> str:
    """由请求 ID 和完整块的逻辑序号编码不依赖前驱的块身份。"""
    if type(request_id) is not int or type(block_index) is not int:
        raise TypeError("request_id 和 block_index 必须为非负整数")
    if request_id < 0 or block_index < 0:
        raise ValueError("request_id 和 block_index 必须为非负整数")
    return f"chunk:{request_id}:{block_index}"


def logical_chunk_id(
    request_id: int, block_index: int, inherited_prefix_ids: tuple[str, ...]
) -> str:
    """按逻辑页位置从不可变继承前缀或请求 ID 推导页身份。"""
    if type(request_id) is not int or type(block_index) is not int:
        raise TypeError("request_id 和 block_index 必须为非负整数")
    if request_id < 0 or block_index < 0:
        raise ValueError("request_id 和 block_index 必须为非负整数")
    if block_index < len(inherited_prefix_ids):
        return inherited_prefix_ids[block_index]  # 沿用父页身份才能命中缓存
    return encode(request_id, block_index)  # 新页归当前请求


def stable_derive(seed: int, domain: str) -> int:
    """按域派生独立随机流，避免采样维度之间相互扰动。"""
    payload = json.dumps([seed, domain], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
