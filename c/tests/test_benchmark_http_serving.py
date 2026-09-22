"""HTTP benchmark contract tests; no model, GPU or third-party packages."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tools import benchmark_http_serving as bench
from openai_server import APIServer


def event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def stream(delta=None, usage=3, finish=True, done=True):
    text = event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    text += event({"choices": [{"index": 0, "delta": delta or {}}]})
    if finish:
        text += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        text += event({"choices": [], "usage": {"completion_tokens": usage}})
    if done:
        text += event("[DONE]")
    return text.encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.payloads.append(body)
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
        try:
            if self.server.barrier:
                self.server.barrier.wait(timeout=5)
            self.send_response(self.server.status)
            self.send_header("Content-Type", self.server.content_type)
            if self.server.status == 302:
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(self.server.body)
        finally:
            with self.server.lock:
                self.server.active -= 1


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.lock = threading.Lock()
        self.server.payloads = []
        self.server.active = self.server.peak = 0
        self.server.status = 200
        self.server.content_type = "text/event-stream"
        self.server.body = stream({"content": "hello"})
        self.server.barrier = None
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        self.workload = [{"messages": [{"role": "user", "content": "hello"}]}]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self):
        return bench.request_one(self.url, self.workload[0], "", 2, 0, time.perf_counter())

    def test_success_and_first_output(self):
        for delta in ({"content": "a"}, {"reasoning_content": "b"},
                      {"reasoning": "c"}, {"tool_calls": [{"function": {"arguments": "{}"}}]}):
            with self.subTest(delta=delta):
                self.server.body = stream(delta)
                row = self.request()
                self.assertTrue(row["success"])
                self.assertEqual(row["completion_tokens"], 3)
                self.assertLessEqual(row["first_output_seconds"], row["duration_seconds"])

    def test_empty_output_is_not_first_output(self):
        self.server.body = stream({"content": "", "tool_calls": [{"id": "id", "function": {}}]}, usage=0)
        row = self.request()
        self.assertTrue(row["success"])
        self.assertIsNone(row["first_output_seconds"])
        self.assertEqual(row["completion_tokens"], 0)

    def test_failures_are_not_successes(self):
        bodies = [stream(done=False), stream(finish=False), b"data: {bad}\n\n",
                  event({"error": {"message": "private diagnostic"}}).encode(),
                  stream(usage=-1), stream(usage=True), b"data: [DONE]"]
        for body in bodies:
            with self.subTest(body=body):
                self.server.body = body
                row = self.request()
                self.assertFalse(row["success"])
                self.assertIsNotNone(row["error"])
                self.assertNotIn("private diagnostic", json.dumps(row))

    def test_http_errors_and_redirects(self):
        for status in (429, 500, 302):
            self.server.status = status
            row = self.request()
            self.assertFalse(row["success"])
            self.assertEqual(row["http_status"], status)
        self.assertEqual(len(self.server.payloads), 3)

    def test_content_type(self):
        self.server.content_type = "application/json"
        self.assertFalse(self.request()["success"])

    def test_missing_usage_disables_token_rate(self):
        self.server.body = stream({"content": "a"}, usage=None)
        row = self.request()
        summary = bench.summarize([row], 1)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["successful_requests_with_usage"], 0)
        self.assertIsNone(summary["successful_completion_tokens_per_second"])

    def test_partial_failure_accounting(self):
        success = self.request()
        failure = dict(success, success=False, completion_tokens=999)
        summary = bench.summarize([success, failure], 2)
        self.assertEqual(summary["failure_rate"], .5)
        self.assertEqual(summary["successful_completion_tokens_per_second"], 1.5)
        self.assertEqual(summary["successful_duration_seconds"]["count"], 1)
        self.assertIsNone(bench.summarize([failure], 1)["successful_completion_tokens_per_second"])

    def test_concurrency_and_payload(self):
        self.server.barrier = threading.Barrier(2)
        rows, summary = bench.run(self.url, self.workload, "test-model", 2, 4, 8, 0, "", 2)
        self.assertEqual(summary["succeeded"], 4)
        self.assertEqual(self.server.peak, 2)
        self.assertEqual([r["index"] for r in rows], list(range(4)))
        for body in self.server.payloads:
            self.assertEqual(body, dict(self.workload[0], model="test-model", stream=True,
                                       stream_options={"include_usage": True}, max_tokens=8,
                                       temperature=0, n=1))

    def test_cli_report_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "prompts.jsonl"
            output = Path(directory) / "report.json"
            workload.write_text(json.dumps(self.workload[0]) + "\n", encoding="utf-8")
            command = [sys.executable, bench.__file__, "--base-url", self.url.rsplit("/", 1)[0],
                       "--model", "fixture", "--workload", str(workload), "--output", str(output)]
            for status, exit_code in ((200, 0), (503, 1)):
                self.server.status = status
                completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, exit_code, completed.stderr)
                report = json.loads(output.read_text())
                self.assertEqual(report["summary"]["succeeded"], 1 - exit_code)
                self.assertEqual(len(report["config"]["workload_sha256"]), 64)
                self.assertNotIn("messages", report["config"])


class ColibriIntegrationTest(unittest.TestCase):
    def test_real_gateway_with_fake_engine(self):
        class Engine:
            def generate(self, prompt, maximum, temperature, top_p, on_text,
                         cache_slot=0, cancelled=None, **kwargs):
                kwargs["on_accept"]({"prompt_tokens": 7})
                on_text("Hello")
                return {"prompt_tokens": 7, "completion_tokens": 1, "length_limited": False}

        server = APIServer(("127.0.0.1", 0), Engine(), "fixture", api_key="test-secret")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            rows, summary = bench.run(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                [{"messages": [{"role": "user", "content": "Hi"}]}],
                "fixture", 1, 2, 8, 0, "test-secret", 2)
            self.assertEqual(summary["succeeded"], 2, rows)
            self.assertEqual(summary["reported_successful_completion_tokens"], 2)
            self.assertEqual(summary["successful_first_output_seconds"]["count"], 2)
            self.assertNotIn("test-secret", json.dumps(rows))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class ParsingTest(unittest.TestCase):
    def test_sse_multiline_comments_crlf_and_partial_eof(self):
        raw = b': ping\r\nevent: message\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: truncated'
        self.assertEqual(list(bench.sse_events(io.BytesIO(raw))), ['{"choices":\n[]}'])

    def test_endpoint(self):
        self.assertEqual(bench.endpoint("http://localhost:8000/v1/"), "http://localhost:8000/v1/chat/completions")
        for url in ("file:///v1", "http://key@host/v1", "http://host/v1?key=secret", "http://host/v1#x"):
            with self.assertRaises(ValueError):
                bench.endpoint(url)

    def test_workload_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            for text in ('', '{}', '{"messages":[]}', '{"messages":[{"role":"user","content":3}]}',
                         '{"messages":[{"role":"user","content":"ok"}],"temperature":1}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    bench.load_workload(path)

    def test_nearest_rank_distribution(self):
        self.assertEqual(bench.distribution(list(range(1, 101)))["p95"], 95)
        self.assertIsNone(bench.distribution([])["p95"])


if __name__ == "__main__":
    unittest.main()
