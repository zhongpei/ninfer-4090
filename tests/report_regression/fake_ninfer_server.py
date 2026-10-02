#!/usr/bin/env python3
"""CPU-only protocol fixture. Not an inference benchmark or a model oracle."""
import argparse
import json
import os
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--request-log-jsonl", required=True)
parser.add_argument("--no-thinking", action="store_true")
parser.add_argument("--spec", default="none")
args, _ = parser.parse_known_args()
lock = threading.Lock()
request_number = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *unused):
        pass

    def reply(self, value):
        raw = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.reply({"status": "ok"} if self.path == "/health" else {"data": [{"id": "fixture"}]})

    def do_POST(self):
        global request_number
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = payload["messages"][0]["content"]
        # A special fixture permits testing a fast-but-wrong candidate.
        if os.environ.get("NINFER_FIXTURE_DIVERGE") == "1" and args.spec != "none":
            content += " different"
        value = {"id": f"fixture-{os.getpid()}", "choices": [{"index": 0,
                 "message": {"role": "assistant", "content": content, "reasoning_content": ""},
                 "finish_reason": "stop"}], "usage": {"completion_tokens": 3}}
        with lock:
            request_number += 1
            event = {"event": "request_done", "server_instance_id": f"fixture-{os.getpid()}",
                     "request": {"request_id": request_number, "enable_thinking": not args.no_thinking,
                                 "sampling": {"temperature": 0, "presence_penalty": 0, "frequency_penalty": 0}},
                     "speculative": {"rounds": 2, "accepted_tokens": 1, "drafted_tokens": 2,
                                     "tree": {"rounds": 0, "fallback_rounds": 0, "nodes": 0, "accepted_drafts": 0},
                                     "lookup": {"rounds": 0, "head_skip_rounds": 0}}}
            with open(args.request_log_jsonl, "a", encoding="utf-8") as log:
                log.write(json.dumps(event) + "\n")
        self.reply(value)


server = ThreadingHTTPServer((args.host, args.port), Handler)
signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
try:
    server.serve_forever(poll_interval=.05)
finally:
    server.server_close()
