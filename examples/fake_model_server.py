"""A stand-in for a local model server (llama.cpp's llama-server, vLLM, ...).

Try `tablely serve` without a GPU or a model file:

    tablely serve examples/models.toml
    curl http://127.0.0.1:8800/v1/chat/completions -H 'Content-Type: application/json' \\
         -d '{"model": "coder", "messages": [{"role": "user", "content": "hi"}]}'

Like llama-server it listens at once and answers /health with 503 until its
"model" has loaded (--load seconds). Replies say which process and device
answered, so model swapping is easy to watch.
"""

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--name", default="fake-model")
    parser.add_argument("--load", type=float, default=0.5, help="seconds until the model is 'loaded'")
    parser.add_argument("--fail", action="store_true", help="exit with an error instead of loading")
    args = parser.parse_args()
    if args.fail:
        print(f"{args.name}: could not load model (simulated)", file=sys.stderr, flush=True)
        sys.exit(1)

    ready_at = time.time() + args.load
    device = os.environ.get("TABLELY_DEVICE", "cpu")
    if device == "cuda":
        device = f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"
        if os.environ.get("TABLELY_GPU_SHARE"):
            device += f" (share {os.environ['TABLELY_GPU_SHARE']})"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):  # keep the job log readable
            pass

        def _json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if time.time() < ready_at:
                self._json(503, {"error": {"message": "Loading model"}})
            elif self.path in ("/health", "/v1/models"):
                self._json(200, {"status": "ok", "data": [{"id": args.name, "object": "model"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if time.time() < ready_at:
                return self._json(503, {"error": {"message": "Loading model"}})
            if self.path == "/v1/embeddings":
                return self._json(200, {"object": "list", "model": body.get("model"),
                                        "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 4}]})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            question = (body.get("messages") or [{}])[-1].get("content", "")
            text = f"[{args.name} pid {os.getpid()} on {device}] you said: {question}"
            if not body.get("stream"):
                return self._json(200, {
                    "id": "chatcmpl-fake", "object": "chat.completion", "model": body.get("model"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}],
                })
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for word in text.split(" "):
                chunk = {"object": "chat.completion.chunk", "model": body.get("model"),
                         "choices": [{"index": 0, "delta": {"content": word + " "}}]}
                self._chunk(f"data: {json.dumps(chunk)}\n\n".encode())
                time.sleep(0.01)
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")

        def _chunk(self, data):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"{args.name}: listening on {args.port}, device {device}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
