#!/usr/bin/env python3
"""
Tests for the eval harness itself.

    python3 -m unittest test_evals -v

An eval harness that scores wrong is worse than none: it produces confident
numbers that point the wrong way. So the scoring rules — what counts as a
hit, what gets skipped, what an argument problem is — are tested against a
scripted fake model, and the HTTP client is tested against a fake Ollama
server, without needing a real model.
"""

import http.server
import json
import pathlib
import tempfile
import threading
import unittest

from evals import run_evals as ev

SCHEMAS = [
    {"type": "function", "function": {
        "name": "web_search",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "search_papers",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "source": {"type": "string",
                                                 "enum": ["arxiv", "semantic_scholar"]}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "old_string": {"type": "string"},
                                      "new_string": {"type": "string"},
                                      "replace_all": {"type": "boolean"}},
                       "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]
INDEX = ev.schema_index(SCHEMAS)


def tool_msg(name, args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def text_msg(text="hello"):
    return {"role": "assistant", "content": text}


def scripted(answers: dict):
    """A fake chat(): maps the user prompt to a reply (or a list, one per call)."""
    calls = {}

    def chat(messages, schemas):
        prompt = messages[-1]["content"]
        reply = answers[prompt]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, list):
            i = calls.get(prompt, 0)
            calls[prompt] = i + 1
            reply = reply[i]
        return reply, {"prompt_s": 0.1, "gen_s": 0.5, "gen_tokens": 10, "prompt_tokens": 900}

    return chat


class TestFirstAction(unittest.TestCase):

    def test_no_tool_calls_means_none(self):
        self.assertEqual(ev.first_action(text_msg()), (ev.NO_TOOL, {}))

    def test_only_the_first_call_counts(self):
        msg = {"tool_calls": [{"function": {"name": "a", "arguments": {}}},
                              {"function": {"name": "b", "arguments": {}}}]}
        self.assertEqual(ev.first_action(msg)[0], "a")

    def test_string_arguments_are_parsed(self):
        """Some models send arguments as a JSON string; the agent accepts that."""
        name, args = ev.first_action(tool_msg("web_search", '{"query": "x"}'))
        self.assertEqual(args, {"query": "x"})

    def test_garbage_string_arguments_are_kept_for_reporting(self):
        name, args = ev.first_action(tool_msg("web_search", "not json"))
        self.assertIn("__unparseable__", args)


class TestCheckArgs(unittest.TestCase):

    def test_valid_call_has_no_problems(self):
        self.assertEqual(ev.check_args("web_search", {"query": "x"}, INDEX), [])

    def test_missing_required(self):
        self.assertIn("missing required 'query'", ev.check_args("web_search", {}, INDEX))

    def test_invented_argument(self):
        problems = ev.check_args("web_search", {"query": "x", "limit": 5}, INDEX)
        self.assertTrue(any("unknown argument 'limit'" in p for p in problems))

    def test_wrong_type(self):
        problems = ev.check_args("edit_file", {"path": "a", "old_string": "b",
                                               "new_string": "c", "replace_all": "yes"}, INDEX)
        self.assertTrue(any("replace_all" in p and "boolean" in p for p in problems))

    def test_enum_violation(self):
        problems = ev.check_args("search_papers", {"query": "x", "source": "google"}, INDEX)
        self.assertTrue(any("not in" in p for p in problems))

    def test_bool_is_not_accepted_as_a_number(self):
        idx = {"t": {"properties": {"n": {"type": "integer"}}, "required": []}}
        self.assertTrue(ev.check_args("t", {"n": True}, idx))

    def test_hallucinated_tool(self):
        self.assertEqual(ev.check_args("calculator", {}, INDEX), ["unknown tool 'calculator'"])

    def test_none_and_error_have_no_arg_problems(self):
        self.assertEqual(ev.check_args(ev.NO_TOOL, {}, INDEX), [])
        self.assertEqual(ev.check_args(ev.ERROR, {}, INDEX), [])


class TestRun(unittest.TestCase):

    def case(self, cid, prompt, expect, category="c"):
        return {"id": cid, "category": category, "prompt": prompt, "expect": expect}

    def test_hit_miss_and_accuracy(self):
        cases = [self.case("a", "news?", ["web_search"]),
                 self.case("b", "papers?", ["search_papers"])]
        chat = scripted({"news?": tool_msg("web_search", {"query": "n"}),
                         "papers?": tool_msg("web_search", {"query": "p"})})
        report = ev.run(cases, chat, SCHEMAS, "sys", progress=None)
        hits = {r["id"]: r["hit"] for r in report["cases"]}
        self.assertEqual(hits, {"a": True, "b": False})
        self.assertEqual(report["summary"]["accuracy"], 0.5)
        self.assertIn(("search_papers -> web_search", 1), report["summary"]["top_confusions"])

    def test_any_accepted_alternative_is_a_hit(self):
        cases = [self.case("x", "fix typo", ["read_file", "edit_file"])]
        chat = scripted({"fix typo": tool_msg("edit_file", {"path": "a", "old_string": "b",
                                                            "new_string": "c"})})
        self.assertTrue(ev.run(cases, chat, SCHEMAS, "sys", progress=None)["cases"][0]["hit"])

    def test_answering_without_tool_scores_against_none(self):
        cases = [self.case("o", "hi", ["none"]), self.case("w", "weather", ["web_search"])]
        chat = scripted({"hi": text_msg(), "weather": text_msg("It is sunny.")})
        hits = {r["id"]: r["hit"] for r in ev.run(cases, chat, SCHEMAS, "sys", progress=None)["cases"]}
        self.assertEqual(hits, {"o": True, "w": False})

    def test_case_skipped_when_primary_tool_missing(self):
        """Skipped cases leave the denominator: a smaller tool set can't win by
        dodging questions it has no tool for."""
        cases = [self.case("q", "prices", ["get_prices"]),
                 self.case("w", "news", ["web_search"])]
        chat = scripted({"news": tool_msg("web_search", {"query": "x"})})
        report = ev.run(cases, chat, SCHEMAS, "sys", progress=None)
        self.assertEqual(report["skipped"], ["q"])
        self.assertEqual(report["summary"]["n"], 1)
        self.assertEqual(report["summary"]["accuracy"], 1.0)

    def test_every_case_skipped_still_summarises(self):
        """--only market in general mode scores nothing; that must print a
        helpful line, not crash on a half-empty summary."""
        report = ev.run([self.case("q", "prices", ["get_prices"])], scripted({}),
                        SCHEMAS, "sys", progress=None)
        report["meta"] = {"mode": "general"}
        self.assertEqual(report["summary"]["n"], 0)
        self.assertIn("No cases scored", ev.format_summary(report))

    def test_unavailable_alternatives_are_dropped_not_skipped(self):
        cases = [self.case("t", "days to xmas", ["web_search", "get_current_time"])]
        chat = scripted({"days to xmas": tool_msg("web_search", {"query": "x"})})
        report = ev.run(cases, chat, SCHEMAS, "sys", progress=None)
        self.assertEqual(report["cases"][0]["expect"], ["web_search"])

    def test_request_error_is_recorded_not_raised(self):
        cases = [self.case("e", "boom", ["web_search"]), self.case("w", "ok", ["web_search"])]
        chat = scripted({"boom": RuntimeError("connection refused"),
                         "ok": tool_msg("web_search", {"query": "x"})})
        report = ev.run(cases, chat, SCHEMAS, "sys", progress=None)
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["cases"][0]["answers"][0]["got"], ev.ERROR)
        self.assertTrue(report["cases"][1]["hit"])

    def test_multiple_runs_measure_consistency(self):
        cases = [self.case("f", "flaky", ["web_search"])]
        chat = scripted({"flaky": [tool_msg("web_search", {"query": "x"}), text_msg()]})
        report = ev.run(cases, chat, SCHEMAS, "sys", runs=2, progress=None)
        rec = report["cases"][0]
        self.assertEqual(rec["hit_rate"], 0.5)
        self.assertFalse(rec["hit"])          # strict: must hit every run
        self.assertFalse(rec["consistent"])
        self.assertEqual(report["summary"]["consistency"], 0.0)

    def test_arg_validity_counts_only_tool_calls(self):
        cases = [self.case("g", "good", ["web_search"]), self.case("b", "bad", ["web_search"]),
                 self.case("n", "hi", ["none"])]
        chat = scripted({"good": tool_msg("web_search", {"query": "x"}),
                         "bad": tool_msg("web_search", {}),
                         "hi": text_msg()})
        report = ev.run(cases, chat, SCHEMAS, "sys", progress=None)
        self.assertEqual(report["summary"]["arg_validity"], 0.5)

    def test_system_prompt_is_sent(self):
        seen = {}

        def chat(messages, schemas):
            seen["system"] = messages[0]
            return text_msg(), {}

        ev.run([self.case("o", "hi", ["none"])], chat, SCHEMAS, "SYSTEM-X", progress=None)
        self.assertEqual(seen["system"], {"role": "system", "content": "SYSTEM-X"})


class TestCompare(unittest.TestCase):

    def report(self, outcomes, model="m", prompt_sha="p"):
        cases = [{"id": i, "hit": hit, "hit_rate": float(hit), "expect": ["web_search"],
                  "answers": [{"got": "web_search" if hit else "none"}]}
                 for i, hit in outcomes.items()]
        return {"cases": cases, "meta": {"model": model, "cases_sha": "c", "prompt_sha": prompt_sha},
                "summary": {"latency_s": {"median": 1.0}}}

    def test_reports_fixes_and_regressions(self):
        out = ev.compare(self.report({"a": False, "b": True}), self.report({"a": True, "b": False}))
        self.assertIn("fixed (1)", out)
        self.assertIn("REGRESSED (1)", out)

    def test_only_common_cases_are_compared(self):
        out = ev.compare(self.report({"a": True}), self.report({"a": True, "new": False}))
        self.assertIn("common cases 1", out)
        self.assertIn("no case changed outcome", out)

    def test_flags_prompt_change(self):
        out = ev.compare(self.report({"a": True}, prompt_sha="1"),
                         self.report({"a": True}, prompt_sha="2"))
        self.assertIn("system prompt changed", out)


class TestCaseFile(unittest.TestCase):

    def test_shipped_cases_parse(self):
        cases = ev.load_cases()
        self.assertGreaterEqual(len(cases), 40)

    def test_every_expected_tool_exists_somewhere(self):
        """A typo in an expected tool name would silently skip that case forever."""
        from tools import TOOL_SCHEMAS
        known = {s["function"]["name"] for s in TOOL_SCHEMAS}
        known |= {ev.NO_TOOL, "extract_archive"}  # extract_archive: added in v6
        try:
            import modes
            known |= {s["function"]["name"] for s in modes.extra_schemas()}
        except ImportError:
            pass
        for case in ev.load_cases():
            for tool in case["expect"]:
                self.assertIn(tool, known, f"case {case['id']} expects unknown tool {tool!r}")

    def test_duplicate_ids_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            line = json.dumps({"id": "a", "category": "c", "prompt": "p", "expect": ["none"]})
            fh.write(line + "\n" + line + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            ev.load_cases(pathlib.Path(fh.name))

    def test_missing_field_rejected_with_line_number(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            fh.write('# comment\n{"id": "a", "prompt": "p", "expect": ["none"]}\n')
        with self.assertRaisesRegex(ValueError, r":2: missing category"):
            ev.load_cases(pathlib.Path(fh.name))


class FakeOllama(http.server.BaseHTTPRequestHandler):
    """Records the request body and answers like /api/chat."""
    received: list = []
    reject_think = False

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(body)
        if FakeOllama.reject_think and "think" in body:
            payload = json.dumps({"error": "model does not support thinking"}).encode()
            self.send_response(400)
        else:
            payload = json.dumps({
                "message": tool_msg("web_search", {"query": "x"}),
                "prompt_eval_count": 1200, "prompt_eval_duration": 2_000_000_000,
                "eval_count": 20, "eval_duration": 1_000_000_000,
            }).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class TestOllamaClient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}/api/chat"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeOllama.received = []
        FakeOllama.reject_think = False

    def test_request_shape_and_timings(self):
        chat = ev.make_ollama_chat("m", 8192, False, 0.0, 7, url=self.url)
        message, t = chat([{"role": "user", "content": "q"}], SCHEMAS)
        body = FakeOllama.received[0]
        self.assertEqual(body["tools"], SCHEMAS)
        self.assertIs(body["stream"], False)
        self.assertIs(body["think"], False)
        self.assertEqual(body["options"], {"num_ctx": 8192, "seed": 7, "temperature": 0.0})
        self.assertEqual(ev.first_action(message)[0], "web_search")
        self.assertAlmostEqual(t["prompt_s"], 2.0)
        self.assertEqual(t["gen_tokens"], 20)

    def test_model_default_temperature_omits_the_option(self):
        chat = ev.make_ollama_chat("m", 8192, False, None, 7, url=self.url)
        chat([{"role": "user", "content": "q"}], SCHEMAS)
        self.assertNotIn("temperature", FakeOllama.received[0]["options"])

    def test_retries_without_think_when_model_rejects_it(self):
        FakeOllama.reject_think = True
        chat = ev.make_ollama_chat("m", 8192, False, 0.0, 7, url=self.url)
        message, _ = chat([{"role": "user", "content": "q"}], SCHEMAS)
        self.assertEqual(len(FakeOllama.received), 2)
        self.assertNotIn("think", FakeOllama.received[1])
        self.assertEqual(ev.first_action(message)[0], "web_search")
        chat([{"role": "user", "content": "q2"}], SCHEMAS)  # remembered: no third retry
        self.assertEqual(len(FakeOllama.received), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
