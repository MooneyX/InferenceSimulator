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


def stable_derive(seed: int, domain: str) -> int:
    """按域派生独立随机流，避免采样维度之间相互扰动。"""
    payload = json.dumps([seed, domain], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
