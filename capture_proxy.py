"""
Capture proxy v2 — saves every API request (one per iteration) to numbered files.
Handles streaming responses by piping chunks through.

Usage:
  1. python3 /tmp/capture_proxy_v2.py &
  2. ANTHROPIC_BASE_URL=http://127.0.0.1:9999 claude -p "your complex task"
  3. Check /tmp/claude_captures/

Output:
  /tmp/claude_captures/
    iteration_001_request.json    — full request body
    iteration_001_summary.txt     — human-readable summary
    iteration_002_request.json
    iteration_002_summary.txt
    ...
    all_iterations_summary.txt    — overview of all iterations
"""

import http.server
import json
import ssl
import urllib.request
import sys
import os
import threading
import time

ANTHROPIC_API = "https://api.anthropic.com"
CAPTURE_DIR = "/tmp/claude_captures"
PORT = 9999

call_counter = 0
call_lock = threading.Lock()
all_summaries = []


def summarize_content_block(block):
    """One-line summary of a content block."""
    if not isinstance(block, dict):
        return str(block)[:100]
    btype = block.get("type", "?")
    if btype == "text":
        text = block.get("text", "")
        has_reminder = "<system-reminder>" in text
        tag = " [SYSTEM-REMINDER]" if has_reminder else ""
        # For system-reminder blocks, show what kind
        if has_reminder:
            if "MCP Server" in text:
                return f"text: <system-reminder> MCP Server Instructions ({len(text)} chars)"
            elif "skills" in text.lower():
                return f"text: <system-reminder> Available Skills ({len(text)} chars)"
            elif "claudeMd" in text or "CLAUDE.md" in text:
                return f"text: <system-reminder> CLAUDE.md + context ({len(text)} chars)"
            elif "task tools" in text.lower():
                return f"text: <system-reminder> Task tools reminder ({len(text)} chars)"
            else:
                return f"text: <system-reminder> ({len(text)} chars)"
        preview = text[:120].replace("\n", " ")
        return f"text: {preview}{'...' if len(text) > 120 else ''} ({len(text)} chars)"
    elif btype == "tool_use":
        name = block.get("name", "?")
        inp = json.dumps(block.get("input", {}))
        inp_preview = inp[:100] + "..." if len(inp) > 100 else inp
        return f"tool_use: {name}({inp_preview})"
    elif btype == "tool_result":
        tid = block.get("tool_use_id", "?")
        is_err = block.get("is_error", False)
        content = block.get("content", "")
        if isinstance(content, list):
            size = sum(len(c.get("text", "")) for c in content if isinstance(c, dict))
        elif isinstance(content, str):
            size = len(content)
        else:
            size = 0
        has_reminder = "<system-reminder>" in str(content)
        err_tag = " ERROR" if is_err else ""
        rem_tag = " +[SYSTEM-REMINDER]" if has_reminder else ""
        return f"tool_result: id={tid}{err_tag} ({size} chars){rem_tag}"
    elif btype == "thinking":
        text = block.get("thinking", "")
        preview = text[:100].replace("\n", " ")
        return f"thinking: {preview}{'...' if len(text) > 100 else ''} ({len(text)} chars)"
    else:
        return f"{btype}: ({len(json.dumps(block))} chars)"


def write_summary(iteration, data, path):
    """Write a human-readable summary of one API request."""
    lines = []
    lines.append(f"{'=' * 70}")
    lines.append(f"ITERATION {iteration}")
    lines.append(f"{'=' * 70}")
    lines.append("")

    # System prompt info
    system = data.get("system", "")
    if isinstance(system, list):
        total = sum(len(b.get("text", "")) for b in system if isinstance(b, dict))
        lines.append(f"System prompt: {len(system)} blocks, {total} chars total")
        for i, block in enumerate(system):
            text = block.get("text", "") if isinstance(block, dict) else str(block)
            cache = block.get("cache_control", None)
            lines.append(f"  [{i}] {len(text)} chars (cache={cache})")
    elif isinstance(system, str):
        lines.append(f"System prompt: {len(system)} chars")
    lines.append("")

    # Model + metadata
    lines.append(f"Model: {data.get('model', '?')}")
    lines.append(f"Max tokens: {data.get('max_tokens', '?')}")
    lines.append(f"Stream: {data.get('stream', '?')}")
    tools = data.get("tools", [])
    lines.append(f"Tools: {len(tools)} defined")
    lines.append("")

    # Messages
    messages = data.get("messages", [])
    lines.append(f"Messages: {len(messages)} total")
    lines.append(f"{'-' * 50}")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", [])
        if isinstance(content, list):
            lines.append(f"  [{i}] role={role} ({len(content)} blocks)")
            for j, block in enumerate(content):
                summary = summarize_content_block(block)
                lines.append(f"       [{j}] {summary}")
        elif isinstance(content, str):
            lines.append(f"  [{i}] role={role} (string, {len(content)} chars)")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_sse_events(raw_sse):
    """Parse SSE text into a list of parsed JSON events."""
    events = []
    for line in raw_sse.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            payload = line[6:]
            if payload == "[DONE]":
                continue
            try:
                events.append(json.loads(payload))
            except json.JSONDecodeError:
                events.append({"_raw": payload})
    return events


def summarize_sse_response(events):
    """Build a human-readable summary of SSE events."""
    lines = []
    text_parts = []
    thinking_parts = []
    tool_uses = []
    usage = None
    stop_reason = None

    for event in events:
        etype = event.get("type", "?")
        if etype == "message_start":
            msg = event.get("message", {})
            lines.append(f"Model: {msg.get('model', '?')}")
            lines.append(f"Message ID: {msg.get('id', '?')}")
            usage = msg.get("usage", {})
        elif etype == "content_block_start":
            block = event.get("content_block", {})
            btype = block.get("type", "?")
            if btype == "tool_use":
                tool_uses.append({
                    "id": block.get("id", "?"),
                    "name": block.get("name", "?"),
                    "input": "",
                })
            elif btype == "thinking":
                pass  # thinking deltas collected below
        elif etype == "content_block_delta":
            delta = event.get("delta", {})
            dtype = delta.get("type", "?")
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

    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts)

    if full_thinking:
        lines.append(f"\nThinking ({len(full_thinking)} chars):")
        preview = full_thinking[:500]
        lines.append(f"  {preview}{'...' if len(full_thinking) > 500 else ''}")

    if full_text:
        lines.append(f"\nText output ({len(full_text)} chars):")
        preview = full_text[:1000]
        lines.append(f"  {preview}{'...' if len(full_text) > 1000 else ''}")

    if tool_uses:
        lines.append(f"\nTool calls ({len(tool_uses)}):")
        for t in tool_uses:
            inp_preview = t["input"][:200] + "..." if len(t["input"]) > 200 else t["input"]
            lines.append(f"  - {t['name']} (id={t['id']})")
            lines.append(f"    input: {inp_preview}")

    lines.append(f"\nStop reason: {stop_reason}")

    if usage:
        lines.append(f"\nUsage:")
        for k, v in usage.items():
            if isinstance(v, (int, float)) and v > 0:
                lines.append(f"  {k}: {v}")

    return "\n".join(lines), {
        "text": full_text,
        "thinking": full_thinking,
        "tool_uses": tool_uses,
        "stop_reason": stop_reason,
        "usage": usage,
    }


def save_response(prefix, body, is_sse=False, is_error=False, status=200):
    """Save the API response."""
    try:
        os.makedirs(CAPTURE_DIR, exist_ok=True)

        # Update meta with response timestamp
        meta_path = os.path.join(CAPTURE_DIR, f"{prefix}_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["response_timestamp"] = time.time()
            meta["duration_ms"] = round((meta["response_timestamp"] - meta["request_timestamp"]) * 1000)
            with open(meta_path, "w") as f:
                json.dump(meta, f)

        # Save raw response
        raw_path = os.path.join(CAPTURE_DIR, f"{prefix}_response_raw.txt")
        with open(raw_path, "w") as f:
            f.write(body)

        # Save parsed summary
        summary_path = os.path.join(CAPTURE_DIR, f"{prefix}_response_summary.txt")
        lines = [
            f"{'=' * 70}",
            f"{prefix.upper()} — RESPONSE",
            f"{'=' * 70}",
            "",
        ]

        if is_error:
            lines.append(f"ERROR (status {status}):")
            lines.append(body[:2000])
        elif is_sse:
            events = parse_sse_events(body)
            lines.append(f"SSE events: {len(events)}")
            summary_text, parsed = summarize_sse_response(events)
            lines.append(summary_text)

            # Also save structured parsed response
            parsed_path = os.path.join(CAPTURE_DIR, f"{prefix}_response_parsed.json")
            with open(parsed_path, "w") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
        else:
            # Non-streaming JSON response
            try:
                data = json.loads(body)
                parsed_path = os.path.join(CAPTURE_DIR, f"{prefix}_response_parsed.json")
                with open(parsed_path, "w") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Summarize
                content = data.get("content", [])
                for block in content:
                    btype = block.get("type", "?")
                    if btype == "text":
                        text = block.get("text", "")
                        lines.append(f"Text ({len(text)} chars): {text[:500]}")
                    elif btype == "tool_use":
                        lines.append(f"Tool use: {block.get('name', '?')} (id={block.get('id','?')})")
                usage = data.get("usage", {})
                if usage:
                    lines.append(f"\nUsage:")
                    for k, v in usage.items():
                        if isinstance(v, (int, float)) and v > 0:
                            lines.append(f"  {k}: {v}")
            except json.JSONDecodeError:
                lines.append(f"Non-JSON response ({len(body)} chars):")
                lines.append(body[:2000])

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        print(f"[proxy] {prefix} response saved ({len(body)} chars)", file=sys.stderr)

    except Exception as e:
        print(f"[proxy] Failed to save response for {prefix}: {e}", file=sys.stderr)


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        global call_counter

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        with call_lock:
            call_counter += 1
            iteration = call_counter

        # Parse and save
        try:
            data = json.loads(body)
            os.makedirs(CAPTURE_DIR, exist_ok=True)

            # Classify the request
            system = data.get("system", "")
            if isinstance(system, list):
                sys_len = sum(len(b.get("text", "")) for b in system if isinstance(b, dict))
                sys_text = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            elif isinstance(system, str):
                sys_len = len(system)
                sys_text = system
            else:
                sys_len = 0
                sys_text = ""

            tools_count = len(data.get("tools", []))
            if tools_count == 0 and sys_len == 0:
                call_type = "internal"
            elif "sub-agent" in sys_text or (0 < sys_len < 10000):
                call_type = "subagent"
            else:
                call_type = "agent"

            prefix = f"{call_type}_{iteration:03d}"

            # Save full request JSON
            request_path = os.path.join(CAPTURE_DIR, f"{prefix}_request.json")
            with open(request_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Save human-readable summary
            summary_path = os.path.join(CAPTURE_DIR, f"{prefix}_summary.txt")
            write_summary(iteration, data, summary_path)

            msg_count = len(data.get("messages", []))
            request_time = time.time()

            # Save timestamp metadata
            meta_path = os.path.join(CAPTURE_DIR, f"{prefix}_meta.json")
            with open(meta_path, "w") as f:
                json.dump({"request_timestamp": request_time, "iteration": iteration, "type": call_type, "prefix": prefix}, f)

            print(f"[proxy] [{call_type.upper():>8}] #{iteration}: {msg_count} messages → {prefix}_request.json", file=sys.stderr)

            all_summaries.append({
                "iteration": iteration,
                "messages": msg_count,
                "file": request_path,
                "type": call_type,
                "prefix": prefix,
                "request_timestamp": request_time,
            })

        except Exception as e:
            print(f"[proxy] Parse error on iteration {iteration}: {e}", file=sys.stderr)

        # Forward to real Anthropic API with streaming support
        url = ANTHROPIC_API + self.path
        req = urllib.request.Request(url, data=body, method="POST")
        for key, value in self.headers.items():
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                req.add_header(key, value)

        try:
            # Remove Accept-Encoding to avoid gzip responses we can't parse
            try:
                req.remove_header("Accept-encoding")
            except (KeyError, AttributeError):
                pass
            req.add_header("Accept-Encoding", "identity")
            ctx = ssl.create_default_context()
            resp = urllib.request.urlopen(req, context=ctx, timeout=300)

            self.send_response(resp.status)
            # Forward headers but handle streaming
            is_streaming = False
            for key, value in resp.getheaders():
                lower = key.lower()
                if lower == "transfer-encoding":
                    continue  # we'll handle this ourselves
                if lower == "content-type" and "event-stream" in value:
                    is_streaming = True
                self.send_header(key, value)

            if is_streaming:
                # Stream SSE chunks through while capturing
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                raw_sse = bytearray()
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        self.wfile.write(b"0\r\n\r\n")
                        break
                    raw_sse.extend(chunk)
                    self.wfile.write(f"{len(chunk):x}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                # Save captured response
                save_response(prefix, raw_sse.decode("utf-8", errors="replace"), is_sse=True)
            else:
                response_body = resp.read()
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
                # Save captured response
                save_response(prefix, response_body.decode("utf-8", errors="replace"), is_sse=False)

        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            response_body = e.read()
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            # Save error response
            save_response(prefix, response_body.decode("utf-8", errors="replace"), is_sse=False, is_error=True, status=e.code)
        except Exception as e:
            print(f"[proxy] Forward error: {e}", file=sys.stderr)
            self.send_response(502)
            error_body = f"Proxy error: {e}".encode()
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
            save_response(prefix, str(e), is_sse=False, is_error=True, status=502)

    def log_message(self, format, *args):
        pass


def write_final_summary():
    """Write overview of all iterations on shutdown."""
    if not all_summaries:
        return
    path = os.path.join(CAPTURE_DIR, "all_iterations_summary.txt")
    lines = [
        f"{'=' * 70}",
        f"CAPTURE SUMMARY — {len(all_summaries)} iterations",
        f"{'=' * 70}",
        "",
    ]
    for s in all_summaries:
        lines.append(f"  #{s['iteration']:3d} [{s.get('type','?'):>8}]: {s['messages']:3d} messages → {s.get('prefix','?')}")
    lines.append("")
    lines.append("Message growth per agent type:")
    for agent_type in ["agent", "subagent"]:
        typed = [s for s in all_summaries if s.get("type") == agent_type]
        if len(typed) >= 2:
            lines.append(f"  {agent_type}:")
            for i in range(1, len(typed)):
                prev = typed[i - 1]["messages"]
                curr = typed[i]["messages"]
                delta = curr - prev
                lines.append(f"    #{typed[i]['iteration']}: +{delta} messages (total {curr})")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[proxy] Final summary: {path}", file=sys.stderr)


if __name__ == "__main__":
    import atexit

    # Create timestamped run folder
    run_id = time.strftime("%Y%m%d_%H%M%S")
    CAPTURE_DIR = os.path.join(CAPTURE_DIR, f"run_{run_id}")
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    atexit.register(write_final_summary)

    server = http.server.HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    print(f"[proxy] Listening on http://127.0.0.1:{PORT}", file=sys.stderr)
    print(f"[proxy] Captures will be saved to {CAPTURE_DIR}/", file=sys.stderr)
    print(f"[proxy] Run: ANTHROPIC_BASE_URL=http://127.0.0.1:{PORT} claude -p 'your task'", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
