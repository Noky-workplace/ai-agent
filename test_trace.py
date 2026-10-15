#!/usr/bin/env python3
"""
Tests for the real-use trace log (turn_trace.py) and its miner.

    python3 -m unittest test_trace -v

The privacy rule is the point: a trace keeps the prompt and the shape of the
turn, never argument values, tool results or answers. So the writer is
driven with tool calls whose arguments and results contain a marker string,
and the test fails if that marker reaches the file.
"""

import builtins
import contextlib
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from unittest import mock

import agent
import files
import turn_trace
from evals import mine_traces

SECRET = "SECRET-7f3a"


def tool_call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class TraceTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="trace-test-")
        self.ws = pathlib.Path(self._tmp.name).resolve()
        self._patches = [mock.patch.object(files, "WORKSPACE", self.ws),
                         mock.patch.object(turn_trace, "ENABLED", True)]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    @property
    def path(self):
        return self.ws / turn_trace.TRACE_FILE

    def lines(self):
        return [json.loads(l) for l in self.path.read_text().splitlines()]

    def run_turn(self, replies, registry, prompt="what's new?", mode="general"):
        replies = iter(replies)
        with mock.patch.object(agent, "call_ollama", side_effect=lambda *a, **k: next(replies)), \
                contextlib.redirect_stdout(io.StringIO()):
            agent.run_agent("qwen3:8b", [{"role": "user", "content": prompt}],
                            schemas=[], registry=registry, mode=mode)


class TestWriter(TraceTest):

    def test_fields_and_outcomes(self):
        registry = {
            "web_search": lambda **kw: f"results about {SECRET}",
            "write_file": lambda **kw: "DENIED: the user declined this change.",
            "read_file": lambda **kw: f"Error: no such file: {SECRET}",
        }
        self.run_turn([
            tool_call("web_search", query=f"news {SECRET}"),
            tool_call("write_file", path=f"{SECRET}.txt", content=SECRET),
            tool_call("read_file", path=SECRET),
            {"role": "assistant", "content": f"answer with {SECRET}"},
        ], registry, mode="finance")
        [t] = self.lines()
        self.assertEqual(t["mode"], "finance")
        self.assertEqual(t["model"], "qwen3:8b")
        self.assertEqual(t["prompt"], "what's new?")
        self.assertEqual(t["calls"], [
            {"tool": "web_search", "arg_keys": ["query"], "outcome": "ok"},
            {"tool": "write_file", "arg_keys": ["content", "path"], "outcome": "denied"},
            {"tool": "read_file", "arg_keys": ["path"], "outcome": "error"},
        ])
        self.assertEqual(t["steps"], 4)
        self.assertFalse(t["max_iterations_hit"])
        self.assertFalse(t["no_tool_calls"])
        self.assertTrue(t["ts"])

    def test_never_logs_argument_values_results_or_answers(self):
        registry = {"web_search": lambda **kw: f"result {SECRET}"}
        self.run_turn([tool_call("web_search", query=SECRET),
                       {"role": "assistant", "content": SECRET}], registry)
        self.assertNotIn(SECRET, self.path.read_text())

    def test_no_tool_turn(self):
        self.run_turn([{"role": "assistant", "content": "hi"}], {})
        [t] = self.lines()
        self.assertTrue(t["no_tool_calls"])
        self.assertEqual(t["calls"], [])

    def test_max_iterations_recorded(self):
        registry = {"web_search": lambda **kw: "r"}
        self.run_turn([tool_call("web_search", query="x")] * agent.MAX_ITERATIONS, registry)
        [t] = self.lines()
        self.assertTrue(t["max_iterations_hit"])
        self.assertEqual(t["steps"], agent.MAX_ITERATIONS)

    def test_ollama_error_still_writes_a_line(self):
        def down(*a, **k):
            raise urllib.error.URLError("refused")
        with mock.patch.object(agent, "call_ollama", side_effect=down), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            agent.run_agent("m", [{"role": "user", "content": "hi"}], schemas=[], registry={})
        [t] = self.lines()
        self.assertTrue(t["ollama_error"])

    def test_disabled_writes_nothing(self):
        with mock.patch.object(turn_trace, "ENABLED", False):
            self.run_turn([{"role": "assistant", "content": "hi"}], {})
        self.assertFalse(self.path.exists())

    def test_no_trace_flag_disables_the_log(self):
        with mock.patch("sys.argv", ["agent.py", "--no-trace"]), \
                mock.patch.object(builtins, "input", side_effect=EOFError), \
                contextlib.redirect_stdout(io.StringIO()):
            agent.main()
        self.assertFalse(turn_trace.ENABLED)

    def test_trace_file_is_protected_from_file_tools(self):
        with self.assertRaises(files.Refused):
            files.contain(turn_trace.TRACE_FILE)


def turn(prompt, calls=(), **flags):
    record = {"ts": "2026-10-15T09:00:00", "mode": "general", "model": "m", "prompt": prompt,
              "calls": [{"tool": t, "arg_keys": [], "outcome": o} for t, o in calls],
              "steps": 1, "max_iterations_hit": False, "no_tool_calls": not calls,
              "ollama_error": False}
    record.update(flags)
    return json.dumps(record, ensure_ascii=False)


class TestMiner(unittest.TestCase):

    def mined(self, *lines):
        return mine_traces.mine(lines)

    def test_each_trigger(self):
        cases = self.mined(
            turn("read my plan", [("read_file", "error")]),
            turn("save this", [("write_file", "denied")]),
            turn("loop forever", [("web_search", "ok")], max_iterations_hit=True),
            turn("what's the latest Python?"),
            turn("今日恒指點？"),
        )
        notes = [c["note"] for c in cases]
        self.assertEqual(len(cases), 5)
        self.assertIn("tool error", notes[0])
        self.assertIn("denied", notes[1])
        self.assertIn("max iterations", notes[2])
        self.assertIn("latest", notes[3])
        self.assertIn("今日", notes[4])

    def test_quiet_turns_are_not_candidates(self):
        self.assertEqual(self.mined(
            turn("search the web", [("web_search", "ok")]),
            turn("hello"),
            turn("do you know Python?"),          # 'know' must not match 'now'
            turn("what's the price?", ollama_error=True),
        ), [])

    def test_output_is_a_case_with_empty_expect(self):
        [c] = self.mined(turn("current HSI level?"))
        self.assertEqual(set(c), {"id", "category", "prompt", "expect", "note"})
        self.assertEqual(c["expect"], [])
        self.assertEqual(c["prompt"], "current HSI level?")

    def test_duplicates_and_bad_lines_skipped(self):
        cases = self.mined(turn("price of AAPL now?"), "not json",
                           turn("price of AAPL now?"))
        self.assertEqual(len(cases), 1)

    def test_main_prints_and_writes_no_files(self):
        evals_dir = pathlib.Path(mine_traces.__file__).parent
        before = {p: p.stat().st_mtime_ns for p in evals_dir.rglob("*") if p.is_file()}
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write(turn("latest news?") + "\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(mine_traces.main(["--file", fh.name]), 0)
        self.assertEqual(json.loads(out.getvalue().splitlines()[0])["prompt"], "latest news?")
        after = {p: p.stat().st_mtime_ns for p in evals_dir.rglob("*") if p.is_file()
                 and "__pycache__" not in p.parts}
        self.assertEqual(after, {p: t for p, t in before.items() if "__pycache__" not in p.parts})


if __name__ == "__main__":
    unittest.main(verbosity=2)
