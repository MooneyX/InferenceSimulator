"""离线与服务端共用的 AgentX JSONL 会话扫描、摘要和请求计划编译器。"""

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from simcore.schemas import TracePlan, TraceTurn

_TIME_TOLERANCE_S = 0.001
_MAX_SESSION_BYTES = 256 * 1024**2
_PLAN_ENVELOPE_BYTES = 128


class PlanTooLargeError(ValueError):
    """所选计划超过当前服务可接受的文件大小。"""


def _sessions(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as source:
        line_number = 0
        while line := source.readline(_MAX_SESSION_BYTES + 1):
            line_number += 1
            if len(line) > _MAX_SESSION_BYTES:
                raise ValueError(f"第 {line_number} 行超过单会话大小限制")
            if line.strip():
                try:
                    session = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise ValueError(f"第 {line_number} 行不是有效 JSON: {exc}") from exc
                if not isinstance(session, dict):
                    raise ValueError(f"第 {line_number} 行须为 JSON 对象")
                yield session


def _requests(session: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    flattened: list[tuple[dict[str, Any], str]] = []
    for item in session["requests"]:
        if item.get("type") == "subagent":
            flattened.extend((child, item["agent_id"]) for child in item["requests"])
        else:
            flattened.append((item, "main"))
    return sorted(flattened, key=lambda pair: pair[0]["t"])


def _topology(requests: list[tuple[dict[str, Any], str]]) -> tuple[list[int | None], int]:
    parents: list[int | None] = []
    ambiguous = 0
    by_lane: dict[str, list[int]] = {}
    for index, (child, lane) in enumerate(requests):
        think = child.get("think_time")
        candidates = (
            [
                predecessor
                for predecessor in by_lane.get(lane, [])
                if requests[predecessor][0]["t"] < child["t"]
                and abs(
                    requests[predecessor][0]["t"]
                    + requests[predecessor][0]["api_time"]
                    + think
                    - child["t"]
                )
                <= _TIME_TOLERANCE_S
            ]
            if think is not None
            else []
        )
        parents.append(candidates[0] if len(candidates) == 1 else None)
        ambiguous += len(candidates) > 1
        by_lane.setdefault(lane, []).append(index)
    return parents, ambiguous


def _inspect(
    session: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], str]], list[int | None], dict[str, Any]]:
    if session.get("hash_id_scope") != "local" or session.get("block_size") != 64:
        raise ValueError("仅支持 64-token 块及会话局部块 ID 的 trace")
    requests = _requests(session)
    if not requests:
        raise ValueError("会话没有推理请求")
    for request, _ in requests:
        if any(
            not isinstance(request.get(name), (int, float))
            or not math.isfinite(request[name])
            or request[name] < 0
            for name in ("t", "api_time")
        ) or (
            "think_time" in request
            and (
                not isinstance(request["think_time"], (int, float))
                or not math.isfinite(request["think_time"])
                or request["think_time"] < 0
            )
        ):
            raise ValueError("请求的原始时间或思考时间无效")
    parents, ambiguous = _topology(requests)
    seen: dict[int, tuple[int, int | None]] = {}
    set_hits = prefix_hits = inputs = outputs = 0
    for request, _ in requests:
        prompt = request["in"]
        output = request["out"]
        hashes = request["hash_ids"]
        if prompt < 1 or output < 0 or len(hashes) != prompt // 64:
            raise ValueError("请求的时间、长度或块 ID 数量无效")
        previous: int | None = None
        prefix_open = True
        for position, block_id in enumerate(hashes):
            if not isinstance(block_id, int) or block_id < 0:
                raise ValueError("块 ID 必须是非负整数")
            prior = seen.get(block_id)
            if prior is not None:
                if prior != (position, previous):
                    raise ValueError("同一会话的块 ID 指向不同输入上下文")
                set_hits += 1
                if prefix_open:
                    prefix_hits += 1
            else:
                prefix_open = False
                seen[block_id] = (position, previous)
            previous = block_id
        inputs += prompt
        outputs += output
    start = requests[0][0]["t"]
    end = requests[-1][0]["t"]
    summary = {
        "id": session["id"],
        "requests": len(requests),
        "subagent_groups": sum(item.get("type") == "subagent" for item in session["requests"]),
        "input_tokens": inputs,
        "output_tokens": outputs,
        "start_s": start,
        "end_s": end,
        "open_loop_qps": (len(requests) - 1) / (end - start) if end > start else None,
        "set_hit_rate": set_hits * 64 / inputs,
        "prefix_hit_rate": prefix_hits * 64 / inputs,
        "inferred_dependencies": sum(parent is not None for parent in parents),
        "timed_roots": sum(parent is None for parent in parents),
        "ambiguous_dependencies": ambiguous,
    }
    return requests, parents, summary


def _plan_turns(
    requests: list[tuple[dict[str, Any], str]],
    parents: list[int | None],
    *,
    session_id: int = 0,
    base: int = 0,
) -> Iterator[TraceTurn]:
    origin = requests[0][0]["t"]
    for index, ((item, _), parent) in enumerate(zip(requests, parents, strict=True)):
        yield TraceTurn(
            request_id=base + index,
            session_id=session_id,
            parent_request_id=base + parent if parent is not None else None,
            arrival_offset_ms=(item["t"] - origin) * 1000,
            think_time_ms=(item["think_time"] * 1000 if parent is not None else None),
            prompt_len=item["in"],
            output_len=item["out"],
            hash_ids=tuple(item["hash_ids"]),
        )


def scan_trace(path: Path) -> list[dict[str, Any]]:
    """逐行解析；只保留小型会话摘要，不保留完整轨迹。"""
    summaries: list[dict[str, Any]] = []
    ids: set[str] = set()
    for session in _sessions(path):
        if session["id"] in ids:
            raise ValueError(f"重复会话 ID: {session['id']}")
        ids.add(session["id"])
        requests, parents, summary = _inspect(session)
        summary["estimated_plan_bytes"] = (
            _PLAN_ENVELOPE_BYTES
            + len(json.dumps(session["id"], ensure_ascii=False).encode("utf-8"))
            + 1
            + sum(
                len(turn.model_dump_json().encode("utf-8")) + 1
                for turn in _plan_turns(requests, parents)
            )
        )
        summaries.append(summary)
    if not summaries:
        raise ValueError("trace 不包含会话")
    return summaries


def compile_plan(path: Path, selected: list[str], *, max_bytes: int | None = None) -> TracePlan:
    """仅为选中的会话物化请求块，逐条限制服务器导出体积。"""
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("须选择至少一个且不重复的会话")
    session_ids: list[str] = []
    turns: list[TraceTurn] = []
    estimated_bytes = _PLAN_ENVELOPE_BYTES
    wanted = set(selected)
    for session in _sessions(path):
        if session["id"] not in wanted:
            continue
        if session["id"] in session_ids:
            raise ValueError("trace 包含重复会话 ID")
        requests, parents, _ = _inspect(session)
        sid = len(session_ids)
        session_ids.append(session["id"])
        estimated_bytes += len(json.dumps(session["id"], ensure_ascii=False).encode("utf-8")) + 1
        for turn in _plan_turns(requests, parents, session_id=sid, base=len(turns)):
            if max_bytes is not None:
                estimated_bytes += len(turn.model_dump_json().encode("utf-8")) + 1
                if estimated_bytes > max_bytes:
                    raise PlanTooLargeError("选中的计划超过服务端允许的大小，请减少会话数量")
            turns.append(turn)
        if len(session_ids) == len(wanted):
            break
    if set(session_ids) != wanted:
        raise ValueError(f"未找到会话: {', '.join(sorted(wanted - set(session_ids)))}")
    return TracePlan(block_size=64, sessions=tuple(session_ids), requests=tuple(turns))


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描 AgentX trace 或导出精简请求计划")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--sessions", nargs="+", help="导出所选会话 ID")
    parser.add_argument("--output", type=Path, help="计划或摘要的输出路径（JSON）")
    args = parser.parse_args()
    if args.sessions:
        if args.output is None:
            parser.error("导出计划时必须指定 --output")
        data = compile_plan(args.trace, args.sessions).model_dump_json()
    else:
        data = json.dumps(scan_trace(args.trace), ensure_ascii=False)
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data)


if __name__ == "__main__":
    main()
