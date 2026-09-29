"""稳定的内容寻址 KV 前缀块身份；仅完整块进入共享目录。"""

import hashlib
import json
from collections.abc import Callable, Iterable

from simcore.schemas import PrefixCacheConfig

Hasher = Callable[[bytes], str]


def stable_derive(seed: int, domain: str) -> int:
    """按域派生独立随机流，避免采样维度之间相互扰动。"""
    payload = json.dumps([seed, domain], ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def cache_namespace(config: PrefixCacheConfig) -> bytes:
    """隔离全部影响 KV 内容的模型维度与显式 salt。"""
    fields = (
        config.model_id,
        config.tokenizer_id,
        config.kv_dtype,
        config.adapter_id,
        config.multimodal_transform_id,
        config.cache_salt,
        config.block_size_tokens,
    )
    return json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def next_chunk_hash(
    namespace: bytes,
    previous_hash: str,
    block: Iterable[int],
    *,
    hasher: Hasher = _digest,
) -> str:
    """增量形成下一个完整块的身份，供输出 token 在步末发布。"""
    payload = json.dumps(
        [namespace.hex(), previous_hash, tuple(block)], separators=(",", ":")
    ).encode()
    return hasher(payload)


def chunk_hashes(
    tokens: Iterable[int],
    block_size_tokens: int,
    namespace: bytes,
    *,
    hasher: Hasher = _digest,
) -> tuple[str, ...]:
    """从 ROOT 链式计算完整块 hash；剩余 partial tail 不产生 hash。"""
    if block_size_tokens < 1:
        raise ValueError("block_size_tokens 必须 >= 1")
    values = tuple(tokens)
    previous = "ROOT"
    result: list[str] = []
    for offset in range(0, len(values) - block_size_tokens + 1, block_size_tokens):
        block = values[offset : offset + block_size_tokens]
        previous = next_chunk_hash(namespace, previous, block, hasher=hasher)
        result.append(previous)
    return tuple(result)
