#!/usr/bin/env python3
"""
Process a proxy capture run folder into viz_data.json for the visualizer.

Usage:
    python3 build_viz_data.py claude_captures/run_20260406_171537/
"""

import json
import hashlib
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


def truncate(text, limit=200):
    if not text or len(text) <= limit:
        return text or ""
    return text[:limit] + "..."


def classify_system_reminder(text):
    if "MCP Server" in text:
        return "mcp"
    if "skills" in text.lower() and "Skill tool" in text:
        return "skills"
    if "claudeMd" in text or "CLAUDE.md" in text:
        return "claude-md"
    if "task tools" in text.lower():
        return "task-tools"
    return "other"


def extract_system_prompt_text(system):
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )
    return ""


def extract_system_prompt_text_stable(system):
    """Extract system prompt text, excluding the billing header (block 0)
    which changes every request due to the cch= fingerprint."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            b.get("text", "") for b in system[1:] if isinstance(b, dict)
        )
    return ""


def summarize_block(block):
    if not isinstance(block, dict):
        return {"type": "unknown", "length": 0}
    btype = block.get("type", "unknown")
    if btype == "text":
        text = block.get("text", "")
        if "<system-reminder>" in text:
            return {
                "type": "system-reminder",
                "category": classify_system_reminder(text),
                "length": len(text),
            }
        return {
            "type": "text",
            "preview": truncate(text),
            "length": len(text),
            "full_text": text,
        }
    elif btype == "tool_use":
        inp = block.get("input", {})
        inp_str = json.dumps(inp) if isinstance(inp, dict) else str(inp)
        return {
            "type": "tool_use",
            "name": block.get("name", "?"),
            "id": block.get("id", "?"),
            "input_preview": truncate(inp_str, 300),
            "input_full": inp_str,
        }
    elif btype == "tool_result":
        content = block.get("content", "")
        if isinstance(content, list):
            text = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict)
            )
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id", "?"),
            "is_error": block.get("is_error", False),
            "preview": truncate(text, 300),
            "length": len(text),
            "has_system_reminder": "<system-reminder>" in text,
        }
    elif btype == "thinking":
        text = block.get("thinking", "")
        return {
            "type": "thinking",
            "preview": truncate(text),
            "full_text": text,
            "length": len(text),
        }
    else:
        return {"type": btype, "length": len(json.dumps(block))}


def summarize_message(msg):
    role = msg.get("role", "?")
    content = msg.get("content", [])
    if isinstance(content, str):
        return {
            "role": role,
            "blocks": [{"type": "text", "preview": truncate(content), "length": len(content), "full_text": content}],
        }
    blocks = [summarize_block(b) for b in content if isinstance(b, dict)]
    return {"role": role, "blocks": blocks}


def extract_task_prompt(messages):
    """Extract the actual user task text from messages[0], skipping system-reminders."""
    if not messages:
        return ""
    first = messages[0]
    content = first.get("content", [])
    if isinstance(content, str):
        return content[:500]
    for block in reversed(content if isinstance(content, list) else []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if "<system-reminder>" not in text and text.strip():
                return text[:500]
    return ""


def extract_response(parsed):
    """Extract response fields from response_parsed.json."""
    if not parsed:
        return {"thinking": "", "text": "", "tool_uses": [], "stop_reason": None, "usage": {}}
    return {
        "thinking": parsed.get("thinking", "") or "",
        "text": parsed.get("text", "") or "",
        "tool_uses": [
            {
                "name": t.get("name", "?"),
                "id": t.get("id", "?"),
                "input_preview": truncate(t.get("input", ""), 200),
            }
            for t in (parsed.get("tool_uses", []) or [])
        ],
        "stop_reason": parsed.get("stop_reason"),
        "usage": parsed.get("usage", {}) or {},
    }


def extract_errors(messages_data, prev_messages_count=0):
    """Find tool_result blocks with is_error=true that are NEW in this iteration.

    Only checks messages after prev_messages_count to avoid counting errors
    from previous iterations that are still in the accumulated history.
    """
    errors = []
    # Only look at messages that are new in this iteration
    new_messages = messages_data[prev_messages_count:]
    for msg in new_messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                text = block.get("content", "")
                if isinstance(text, list):
                    text = " ".join(c.get("text", "") for c in text if isinstance(c, dict))
                errors.append({
                    "tool_use_id": block.get("tool_use_id", "?"),
                    "preview": truncate(str(text), 200),
                })
    return errors


def find_agent_tool_uses(messages_data):
    """Find Agent tool_use blocks in the messages (for linking subagents)."""
    agent_calls = []
    for msg in messages_data:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "Agent":
                inp = block.get("input", {})
                prompt = inp.get("prompt", "") if isinstance(inp, dict) else ""
                agent_calls.append({
                    "tool_use_id": block.get("id", "?"),
                    "prompt": prompt,
                    "description": inp.get("description", "") if isinstance(inp, dict) else "",
                    "subagent_type": inp.get("subagent_type", "") if isinstance(inp, dict) else "",
                })
    return agent_calls


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return None


def try_decompress_and_parse_sse(raw_path):
    """Try to decompress a gzip SSE response and parse it."""
    import gzip
    try:
        with open(raw_path, "rb") as f:
            raw = f.read()
        if not raw or raw[:2] != b'\x1f\x8b':
            return None  # not gzip
        decompressed = gzip.decompress(raw).decode("utf-8", errors="replace")
        return parse_sse_text(decompressed)
    except Exception:
        return None


def parse_sse_text(text):
    """Parse SSE text into structured response data."""
    events = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                pass
    if not events:
        return None

    text_parts = []
    thinking_parts = []
    tool_uses = []
    usage = None
    stop_reason = None

    for event in events:
        etype = event.get("type", "")
        if etype == "message_start":
            msg = event.get("message", {})
            usage = msg.get("usage", {})
        elif etype == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                tool_uses.append({
                    "id": block.get("id", "?"),
                    "name": block.get("name", "?"),
                    "input": "",
                })
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "text_delta":
                text_parts.append(delta.get("text", ""))
            elif dtype == "input_json_delta":
                if tool_uses:
                    tool_uses[-1]["input"] += delta.get("partial_json", "")
            elif dtype == "thinking_delta":
                thinking_parts.append(delta.get("thinking", ""))
        elif etype == "message_delta":
            delta = event.get("delta", {})
            stop_reason = delta.get("stop_reason")
            msg_usage = event.get("usage", {})
            if msg_usage:
                usage = {**(usage or {}), **msg_usage}

    # Parse tool use inputs
    for tu in tool_uses:
        try:
            tu["input"] = json.loads(tu["input"]) if tu["input"] else {}
        except json.JSONDecodeError:
            tu["input"] = tu["input"]

    return {
        "text": "".join(text_parts),
        "thinking": "".join(thinking_parts),
        "tool_uses": tool_uses,
        "stop_reason": stop_reason,
        "usage": usage,
    }


def discover_iterations(run_dir):
    """Find all iteration prefixes and their associated files."""
    files = os.listdir(run_dir)
    prefixes = set()
    for f in files:
        m = re.match(r"^((?:agent|subagent|internal)_\d{3})", f)
        if m:
            prefixes.add(m.group(1))

    iterations = []
    for prefix in sorted(prefixes, key=lambda p: int(re.search(r"\d{3}", p).group())):
        num = int(re.search(r"\d{3}", prefix).group())
        call_type = prefix.split("_")[0]
        response = load_json(os.path.join(run_dir, f"{prefix}_response_parsed.json"))

        # If parsed response is empty/null, try decompressing the raw SSE file
        is_empty = not response or (
            not response.get("text") and
            not response.get("tool_uses") and
            not response.get("thinking")
        )
        if is_empty:
            raw_path = os.path.join(run_dir, f"{prefix}_response_raw.txt")
            recovered = try_decompress_and_parse_sse(raw_path)
            if recovered:
                response = recovered

        entry = {
            "prefix": prefix,
            "number": num,
            "type": call_type,
            "request": load_json(os.path.join(run_dir, f"{prefix}_request.json")),
            "response": response,
            "meta": load_json(os.path.join(run_dir, f"{prefix}_meta.json")),
        }
        iterations.append(entry)
    return iterations


def build_viz_data(run_dir):
    run_id = os.path.basename(run_dir.rstrip("/"))
    iterations = discover_iterations(run_dir)

    if not iterations:
        print(f"No iterations found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    # Separate by type
    agent_iters = [i for i in iterations if i["type"] == "agent"]
    subagent_iters = [i for i in iterations if i["type"] == "subagent"]
    internal_iters = [i for i in iterations if i["type"] == "internal"]

    # System prompt analysis
    system_prompts = {}
    for it in iterations:
        req = it.get("request")
        if not req:
            continue
        sys_text = extract_system_prompt_text(req.get("system", ""))
        if sys_text:
            h = md5(sys_text)
            if h not in system_prompts:
                system_prompts[h] = {"hash": h, "length": len(sys_text), "text": sys_text, "types": set()}
            system_prompts[h]["types"].add(it["type"])

    parent_prompt = None
    subagent_prompt = None
    for h, info in system_prompts.items():
        if "agent" in info["types"] and (parent_prompt is None or info["length"] > parent_prompt["length"]):
            parent_prompt = info
        if "subagent" in info["types"] and (subagent_prompt is None or info["length"] > 0):
            subagent_prompt = info

    # Check if system prompt changed between iterations of the same agent type
    # Uses extract_system_prompt_text_stable to ignore billing header changes
    parent_prompt_changed = False
    agent_hashes = set()
    for it in agent_iters:
        req = it.get("request")
        if req:
            sys_text = extract_system_prompt_text_stable(req.get("system", ""))
            if sys_text:
                agent_hashes.add(md5(sys_text))
    if len(agent_hashes) > 1:
        parent_prompt_changed = True

    # Check per subagent group
    subagent_prompt_changed = {}  # group_key -> bool

    # Group subagent iterations into logical agents
    # Key insight: all iterations of the same subagent share the same task prompt
    # (messages[0] last text block) and system prompt, but message count grows.
    # Group by task text only (first 200 chars to handle minor variations).
    subagent_groups = defaultdict(list)
    for it in subagent_iters:
        req = it.get("request")
        if not req:
            continue
        task = extract_task_prompt(req.get("messages", []))
        # Use first 200 chars of task — enough to distinguish agents but
        # ignores trailing content that might vary
        group_key = md5(task[:200])
        subagent_groups[group_key].append(it)

    # Check per-subagent prompt stability (ignoring billing header)
    for gk, group in subagent_groups.items():
        hashes = set()
        for it in group:
            req = it.get("request")
            if req:
                sys_text = extract_system_prompt_text_stable(req.get("system", ""))
                if sys_text:
                    hashes.add(md5(sys_text))
        subagent_prompt_changed[gk] = len(hashes) > 1

    # Link subagents to parent tool_use IDs
    # Search ALL iterations (not just agent) because the Agent tool_use from
    # iteration #001's response appears in the messages of later iterations.
    parent_agent_calls = []
    seen_tool_ids = set()
    for it in iterations:
        req = it.get("request")
        if req:
            calls = find_agent_tool_uses(req.get("messages", []))
            for call in calls:
                if call["tool_use_id"] not in seen_tool_ids:
                    seen_tool_ids.add(call["tool_use_id"])
                    parent_agent_calls.append(call)

    def match_subagent_to_parent(task_text):
        for call in parent_agent_calls:
            if task_text and call["prompt"] and (
                task_text[:100] in call["prompt"] or call["prompt"][:100] in task_text
            ):
                return call
        return None

    # Build agent entries
    agents = []

    # Detect which parent iteration is the fork (spawned agents).
    # The fork is the FIRST parent iteration — its response spawns agents,
    # even if we can't see the response (gzip issue). We detect it by checking
    # if the NEXT parent iteration's messages contain Agent tool_uses + tool_results.
    fork_iteration_num = None
    if agent_iters and subagent_iters:
        # The fork is the parent iteration just before subagents start
        first_sub_num = min(it["number"] for it in subagent_iters)
        for it in agent_iters:
            if it["number"] < first_sub_num:
                fork_iteration_num = it["number"]

    # Parent agent
    parent_iterations = []
    prev_parent_msg_count = 0
    for it in agent_iters:
        req = it.get("request", {}) or {}
        messages = req.get("messages", [])
        resp_data = extract_response(it.get("response"))
        meta = it.get("meta", {}) or {}
        errors = extract_errors(messages, prev_parent_msg_count)

        # Check spawned agents from response tool_uses
        spawned = []
        resp_agent_calls = [t for t in (resp_data.get("tool_uses") or []) if t.get("name") == "Agent"]
        msg_agent_calls = find_agent_tool_uses(messages)
        all_calls = resp_agent_calls + msg_agent_calls

        for call in all_calls:
            prompt = call.get("prompt") or call.get("input_preview", "")
            for gk, group in subagent_groups.items():
                task = extract_task_prompt((group[0].get("request") or {}).get("messages", []))
                if task and prompt and task[:100] in prompt:
                    sid = f"subagent_{gk[:8]}"
                    if sid not in spawned:
                        spawned.append(sid)

        # If this is the fork iteration and we couldn't detect spawns from response,
        # mark all subagents as spawned here
        if it["number"] == fork_iteration_num and not spawned:
            spawned = [f"subagent_{gk[:8]}" for gk in subagent_groups.keys()]

        parent_iterations.append({
            "number": it["number"],
            "prefix": it["prefix"],
            "timestamp_start": meta.get("request_timestamp", 0),
            "timestamp_end": meta.get("response_timestamp", 0),
            "duration_ms": meta.get("duration_ms", 0),
            "messages_count": len(messages),
            "messages": [summarize_message(m) for m in messages],
            "response": resp_data,
            "usage": resp_data.get("usage", {}),
            "errors": errors,
            "spawned_agents": spawned,
            "raw_messages": messages,
            "raw_request": it.get("request"),
            "raw_response": it.get("response"),
        })
        prev_parent_msg_count = len(messages)

    parent_task = ""
    if agent_iters:
        req = agent_iters[0].get("request", {}) or {}
        parent_task = extract_task_prompt(req.get("messages", []))

    parent_tool_chain = []
    for pi in parent_iterations:
        tools = [t["name"] for t in pi["response"].get("tool_uses", [])]
        parent_tool_chain.append({"iteration": pi["number"], "tools": tools})

    agents.append({
        "id": "parent",
        "type": "agent",
        "system_prompt_hash": parent_prompt["hash"] if parent_prompt else "",
        "task": truncate(parent_task, 500),
        "iterations": parent_iterations,
        "tool_chain": parent_tool_chain,
    })

    # Subagent entries
    for gk, group in sorted(subagent_groups.items()):
        group.sort(key=lambda i: i["number"])
        agent_id = f"subagent_{gk[:8]}"
        first_req = (group[0].get("request") or {})
        task = extract_task_prompt(first_req.get("messages", []))
        sys_text = extract_system_prompt_text(first_req.get("system", ""))
        sys_hash = md5(sys_text) if sys_text else ""

        parent_match = match_subagent_to_parent(task)

        sub_iterations = []
        sub_tool_chain = []
        prev_sub_msg_count = 0
        for it in group:
            req = it.get("request", {}) or {}
            messages = req.get("messages", [])
            resp_data = extract_response(it.get("response"))
            meta = it.get("meta", {}) or {}
            errors = extract_errors(messages, prev_sub_msg_count)

            sub_iterations.append({
                "number": it["number"],
                "prefix": it["prefix"],
                "timestamp_start": meta.get("request_timestamp", 0),
                "timestamp_end": meta.get("response_timestamp", 0),
                "duration_ms": meta.get("duration_ms", 0),
                "messages_count": len(messages),
                "messages": [summarize_message(m) for m in messages],
                "response": resp_data,
                "usage": resp_data.get("usage", {}),
                "errors": errors,
                "spawned_agents": [],
                "raw_messages": messages,
                "raw_request": it.get("request"),
                "raw_response": it.get("response"),
            })
            prev_sub_msg_count = len(messages)
            tools = [t["name"] for t in resp_data.get("tool_uses", [])]
            sub_tool_chain.append({"iteration": it["number"], "tools": tools})

        agents.append({
            "id": agent_id,
            "type": "subagent",
            "parent_tool_use_id": parent_match["tool_use_id"] if parent_match else "",
            "parent_description": parent_match["description"] if parent_match else "",
            "task": truncate(task, 500),
            "system_prompt_hash": sys_hash,
            "iterations": sub_iterations,
            "tool_chain": sub_tool_chain,
        })

    # Internal calls
    internal_calls = []
    for it in internal_iters:
        req = it.get("request", {}) or {}
        messages = req.get("messages", [])
        content_preview = ""
        content_length = 0
        if messages:
            c = messages[0].get("content", "")
            if isinstance(c, str):
                content_preview = truncate(c)
                content_length = len(c)
        resp = it.get("response")
        meta = it.get("meta", {}) or {}
        internal_calls.append({
            "number": it["number"],
            "prefix": it["prefix"],
            "content_preview": content_preview,
            "content_length": content_length,
            "response_preview": truncate(json.dumps(resp), 200) if resp else "",
            "timestamp_start": meta.get("request_timestamp", 0),
            "duration_ms": meta.get("duration_ms", 0),
        })

    # Timeline
    timeline = []
    for it in iterations:
        agent_id = "parent" if it["type"] == "agent" else ""
        if it["type"] == "subagent":
            req = it.get("request", {}) or {}
            task = extract_task_prompt(req.get("messages", []))
            sys_text = extract_system_prompt_text(req.get("system", ""))
            gk = md5(task + sys_text)
            agent_id = f"subagent_{gk[:8]}"
        elif it["type"] == "internal":
            agent_id = "internal"
        meta = it.get("meta", {}) or {}
        timeline.append({
            "number": it["number"],
            "type": it["type"],
            "agent_id": agent_id,
            "duration_ms": meta.get("duration_ms", 0),
            "timestamp": meta.get("request_timestamp", 0),
        })

    # System reminders
    system_reminders = []
    for it in iterations:
        req = it.get("request", {}) or {}
        for msg in req.get("messages", []):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if "<system-reminder>" in text:
                        system_reminders.append({
                            "iteration": it["number"],
                            "category": classify_system_reminder(text),
                            "length": len(text),
                        })

    # Token totals
    total_input = 0
    total_output = 0
    total_cache_create = 0
    total_cache_read = 0
    total_duration = 0
    for agent in agents:
        for pit in agent["iterations"]:
            usage = pit.get("usage", {})
            total_input += usage.get("input_tokens", 0)
            total_output += usage.get("output_tokens", 0)
            total_cache_create += usage.get("cache_creation_input_tokens", 0)
            total_cache_read += usage.get("cache_read_input_tokens", 0)
            total_duration += pit.get("duration_ms", 0)

    # Wall clock
    all_timestamps = [t["timestamp"] for t in timeline if t["timestamp"]]
    wall_clock_ms = int((max(all_timestamps) - min(all_timestamps)) * 1000) if len(all_timestamps) >= 2 else 0

    viz_data = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_calls": len(iterations),
            "agent_calls": len(agent_iters),
            "subagent_calls": len(subagent_iters),
            "internal_calls": len(internal_iters),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_creation_tokens": total_cache_create,
            "total_cache_read_tokens": total_cache_read,
            "total_duration_ms": total_duration,
            "wall_clock_ms": wall_clock_ms,
        },
        "system_prompts": {
            "parent": {"hash": parent_prompt["hash"], "length": parent_prompt["length"], "text": parent_prompt["text"], "changed": parent_prompt_changed} if parent_prompt else None,
            "subagent": {"hash": subagent_prompt["hash"], "length": subagent_prompt["length"], "text": subagent_prompt["text"]} if subagent_prompt else None,
            "per_agent": {},  # filled below
            "parent_changed": parent_prompt_changed,
            "subagent_changed": any(subagent_prompt_changed.values()),
            "per_subagent_changed": {f"subagent_{gk[:8]}": v for gk, v in subagent_prompt_changed.items()},
        },
        "agents": agents,
        "internal_calls": internal_calls,
        "timeline": timeline,
        "system_reminders": system_reminders,
    }

    # Populate per_agent system prompts (using stable text, excluding billing header)
    for agent in agents:
        if agent["iterations"]:
            req = agent["iterations"][0].get("raw_request", {})
            sys = req.get("system", [])
            if isinstance(sys, list) and len(sys) > 1:
                # Skip block 0 (billing), include block 1+ as the real prompt
                stable_text = "\n\n".join(b.get("text", "") for b in sys[1:] if isinstance(b, dict))
                viz_data["system_prompts"]["per_agent"][agent["id"]] = {
                    "hash": md5(stable_text),
                    "length": len(stable_text),
                    "text": stable_text,
                    "blocks": [{"length": len(b.get("text","")), "cache_control": b.get("cache_control")} for b in sys],
                }

    # Clean up sets (not JSON serializable)
    for h, info in system_prompts.items():
        info["types"] = list(info["types"])

    return viz_data


def print_summary(viz):
    s = viz["summary"]
    print(f"\n{'='*60}")
    print(f"  Run: {viz['run_id']}")
    print(f"  Calls: {s['total_calls']} ({s['agent_calls']} agent, {s['subagent_calls']} subagent, {s['internal_calls']} internal)")
    print(f"  Tokens: {s['total_input_tokens']:,} input, {s['total_output_tokens']:,} output")
    print(f"  Cache: {s['total_cache_read_tokens']:,} read, {s['total_cache_creation_tokens']:,} created")
    print(f"  Wall clock: {s['wall_clock_ms']/1000:.1f}s")
    print(f"{'='*60}")

    for agent in viz["agents"]:
        prefix = "  " if agent["type"] == "agent" else "    "
        iters = len(agent["iterations"])
        task = truncate(agent.get("task", ""), 80)
        tool_names = []
        for tc in agent.get("tool_chain", []):
            tool_names.extend(tc.get("tools", []))
        tools_str = " → ".join(tool_names[:15])
        if len(tool_names) > 15:
            tools_str += f" ... ({len(tool_names)} total)"
        print(f"{prefix}[{agent['id']}] {iters} iterations — {task}")
        if tools_str:
            print(f"{prefix}  tools: {tools_str}")

    if viz["internal_calls"]:
        print(f"  [{len(viz['internal_calls'])} internal calls]")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <run_folder>", file=sys.stderr)
        sys.exit(1)

    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"Not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)

    viz = build_viz_data(run_dir)
    out_path = os.path.join(run_dir, "viz_data.json")
    with open(out_path, "w") as f:
        json.dump(viz, f, indent=2, ensure_ascii=False)

    print_summary(viz)
    print(f"  Written to: {out_path}")
    print(f"  Size: {os.path.getsize(out_path):,} bytes")
