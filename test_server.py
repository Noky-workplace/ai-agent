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
import urllib.error
import urllib.request

import server


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
