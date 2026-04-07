"""
Minimal HTTP proxy that captures the request body sent to Anthropic,
saves the system prompt to a file, then forwards the request.

Usage:
  1. python3 /tmp/capture_proxy.py &
  2. ANTHROPIC_BASE_URL=http://127.0.0.1:9999 claude -p "say hi"
  3. Check /tmp/captured_system_prompt.txt
"""

import http.server
import json
import ssl
import urllib.request
import sys

ANTHROPIC_API = "https://api.anthropic.com"
CAPTURE_FILE = "/tmp/captured_system_prompt.txt"
FULL_REQUEST_FILE = "/tmp/captured_full_request.json"
PORT = 9999


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Parse and capture the system prompt
        try:
            data = json.loads(body)
            # Save full request (minus messages content for size)
            with open(FULL_REQUEST_FILE, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n[proxy] Captured full request to {FULL_REQUEST_FILE}", file=sys.stderr)

            # Extract and save system prompt specifically
            system = data.get("system")
            if system:
                # system can be a string or list of objects
                if isinstance(system, str):
                    prompt_text = system
                elif isinstance(system, list):
                    parts = []
                    for block in system:
                        if isinstance(block, dict):
                            parts.append(block.get("text", json.dumps(block)))
                        else:
                            parts.append(str(block))
                    prompt_text = "\n\n".join(parts)
                else:
                    prompt_text = str(system)

                with open(CAPTURE_FILE, "w") as f:
                    f.write(prompt_text)
                print(f"[proxy] System prompt saved to {CAPTURE_FILE} ({len(prompt_text)} chars)", file=sys.stderr)
        except Exception as e:
            print(f"[proxy] Parse error: {e}", file=sys.stderr)

        # Forward to real Anthropic API
        url = ANTHROPIC_API + self.path
        req = urllib.request.Request(url, data=body, method="POST")
        for key, value in self.headers.items():
            if key.lower() not in ("host", "content-length"):
                req.add_header(key, value)

        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, context=ctx) as resp:
                self.send_response(resp.status)
                for key, value in resp.getheaders():
                    if key.lower() not in ("transfer-encoding",):
                        self.send_header(key, value)
                response_body = resp.read()
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            response_body = e.read()
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

    def log_message(self, format, *args):
        pass  # Suppress default logging


if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    print(f"[proxy] Listening on http://127.0.0.1:{PORT}", file=sys.stderr)
    print(f"[proxy] Run: ANTHROPIC_BASE_URL=http://127.0.0.1:{PORT} claude -p 'say hi'", file=sys.stderr)
    server.serve_forever()