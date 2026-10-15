#!/usr/bin/env python3
"""
Tests for the web server's turn handling.

    python3 -m unittest test_server -v

A second tab can post while a turn is running. That request must be turned
away before it touches anything: a mode switch that resets STATE["messages"]
mid-turn would drop the running turn's answer from history and leave its tools
having run under the old mode. These run the real Handler on a spare port;
no browser, no Ollama.
"""

import http.server
import json
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request

import server



def setUpModule():
    # These tests drive run_agent; keep them out of the real trace log.
    import turn_trace
    turn_trace.ENABLED = False

class TestBusyTurn(unittest.TestCase):

    def setUp(self):
        server._reset("general")
        server.STATE["status"] = "idle"
        server.Handler.opts = {"model": "fake", "ctx": 16384, "think": False}
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/chat"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        server.STATE["status"] = "idle"

    def post(self, body):
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_mode_switch_while_busy_is_refused_and_changes_nothing(self):
        messages = server.STATE["messages"]
        messages.append({"role": "user", "content": "turn in progress"})
        snapshot = list(messages)
        server.STATE["status"] = "thinking"

        code, body = self.post({"text": "hi", "mode": "finance"})

        self.assertEqual(code, 409, body)
        self.assertIs(server.STATE["messages"], messages)   # same list, not replaced
        self.assertEqual(server.STATE["messages"], snapshot)
        self.assertEqual(server.STATE["view"].name, "general")

    def test_mode_switch_when_idle_resets(self):
        code, _ = self.post({"text": "/clear", "mode": "finance"})
        self.assertEqual(code, 200)
        self.assertEqual(server.STATE["view"].name, "finance")
        self.assertEqual(len(server.STATE["messages"]), 1)

    def test_unknown_mode_rejected(self):
        code, _ = self.post({"text": "hi", "mode": "evil"})
        self.assertEqual(code, 400)
        self.assertEqual(server.STATE["view"].name, "general")


class TestStream(unittest.TestCase):
    """GET /stream forwards what a turn publishes, as server-sent events."""

    def setUp(self):
        server._reset("general")
        server.Handler.opts = {"model": "fake", "ctx": 16384, "think": False}
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    def open_stream(self):
        resp = urllib.request.urlopen(self.base + "/stream", timeout=5)
        self.assertEqual(resp.headers["Content-Type"], "text/event-stream")
        for _ in range(100):                 # wait until the handler registered
            if server.STREAMS:
                return resp
            threading.Event().wait(0.01)
        self.fail("stream never registered")

    def next_event(self, resp):
        while True:
            line = resp.readline().decode("utf-8").strip()
            if line.startswith("data: "):
                return json.loads(line[len("data: "):])

    def test_tokens_and_reset_reach_the_page(self):
        resp = self.open_stream()
        server._on_token("Hel")
        server._on_token(None)
        server._on_token("lo")
        self.assertEqual([self.next_event(resp) for _ in range(3)],
                         [{"type": "token", "text": "Hel"}, {"type": "reset"},
                          {"type": "token", "text": "lo"}])
        resp.close()

    def test_turn_passes_the_stream_callback_to_the_agent(self):
        seen = {}
        def fake_run_agent(*a, **kw):
            seen.update(kw)
            print("agent> done")
        with unittest.mock.patch.object(server.agent, "run_agent", fake_run_agent):
            server._run_turn("hi", server.Handler.opts)
        self.assertIs(seen["on_token"], server._on_token)
        self.assertIs(seen["registry"], server.STATE["view"].registry)


if __name__ == "__main__":
    unittest.main(verbosity=2)
