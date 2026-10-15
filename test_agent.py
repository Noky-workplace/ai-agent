#!/usr/bin/env python3
"""
Tests for mode-aware dispatch in agent.py.

    python3 -m unittest test_agent -v

A mode hides tools by leaving their schemas out of the request. That alone
is not a boundary: a model can still emit a call to a tool it was never
shown, by hallucination or because something it read told it to. The
boundary is that execute_tool dispatches through the mode's registry, so a
hidden tool's name resolves to nothing. These tests pin that down end to end
with a scripted fake model, so no Ollama is needed.
"""

import contextlib
import io
import json
import unittest
from unittest import mock

import agent


def tool_call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}



def setUpModule():
    # These tests drive run_agent; keep them out of the real trace log.
    import turn_trace
    turn_trace.ENABLED = False

class TestModeDispatch(unittest.TestCase):

    def run_scripted(self, view, replies):
        """Run one turn against a fake model; return (messages, call_ollama mock)."""
        replies = iter(replies)
        messages = [{"role": "system", "content": view.prompt},
                    {"role": "user", "content": "go"}]
        with mock.patch.object(agent, "call_ollama",
                               side_effect=lambda *a, **k: next(replies)) as fake, \
                contextlib.redirect_stdout(io.StringIO()):
            agent.run_agent("fake", messages,
                            schemas=view.schemas, registry=view.registry)
        return messages, fake

    def test_hidden_tool_is_refused_even_when_the_model_names_it(self):
        view = agent.build_view("finance")
        self.assertIn("delete_file", agent.TOOL_REGISTRY)
        self.assertNotIn("delete_file", view.registry)

        ran = []
        with mock.patch.dict(agent.TOOL_REGISTRY,
                             {"delete_file": lambda **kw: ran.append(kw) or "deleted"}):
            messages, _ = self.run_scripted(view, [
                tool_call("delete_file", path="README.md"),
                {"role": "assistant", "content": "ok"},
            ])

        self.assertEqual(ran, [])
        result = next(m for m in messages if m["role"] == "tool")
        self.assertIn("no tool named 'delete_file'", result["content"])
        self.assertNotIn("delete_file", result["content"].split("Available:")[1])

    def test_request_carries_only_the_modes_schemas(self):
        view = agent.build_view("finance")
        _, fake = self.run_scripted(view, [{"role": "assistant", "content": "hi"}])
        sent = fake.call_args.args[4]
        self.assertEqual([s["function"]["name"] for s in sent],
                         [s["function"]["name"] for s in view.schemas])
        self.assertNotIn("delete_file", {s["function"]["name"] for s in sent})

    def test_visible_tool_still_runs(self):
        view = agent.build_view("finance")
        with mock.patch.dict(view.registry, {"get_current_time": lambda: "NOON"}):
            messages, _ = self.run_scripted(view, [
                tool_call("get_current_time"),
                {"role": "assistant", "content": "ok"},
            ])
        result = next(m for m in messages if m["role"] == "tool")
        self.assertEqual(result["content"], "NOON")

    def test_execute_tool_without_registry_never_falls_back_to_globals(self):
        """get_prices is hidden in general mode. Planted in the global registry,
        a caller that forgets `registry` must still not reach it."""
        ran = []
        with mock.patch.dict(agent.TOOL_REGISTRY,
                             {"get_prices": lambda **kw: ran.append(kw) or "data"}):
            out = agent.execute_tool("get_prices", {"ticker": "AAPL", "start": "2025-01-01"})
        self.assertEqual(ran, [])
        self.assertIn("no tool named 'get_prices'", out)

    def test_run_agent_without_mode_args_uses_general_mode(self):
        ran = []
        replies = iter([tool_call("get_prices", ticker="AAPL", start="2025-01-01"),
                        {"role": "assistant", "content": "ok"}])
        messages = [{"role": "user", "content": "go"}]
        with mock.patch.dict(agent.TOOL_REGISTRY,
                             {"get_prices": lambda **kw: ran.append(kw) or "data"}), \
                mock.patch.object(agent, "call_ollama",
                                  side_effect=lambda *a, **k: next(replies)) as fake, \
                contextlib.redirect_stdout(io.StringIO()):
            agent.run_agent("fake", messages)
        self.assertEqual(ran, [])
        sent = {s["function"]["name"] for s in fake.call_args.args[4]}
        self.assertEqual(sent, {s["function"]["name"] for s in agent.general_tools()[0]})
        self.assertNotIn("get_prices", sent)

    def test_general_mode_is_the_v6_tool_set(self):
        view = agent.build_view("general")
        self.assertEqual(view.schemas, agent.TOOL_SCHEMAS)
        self.assertEqual(set(view.registry), set(agent.TOOL_REGISTRY))

    def test_finance_mode_has_12_tools(self):
        self.assertEqual(len(agent.build_view("finance").schemas), 12)


def ndjson(*chunks):
    """A fake streamed /api/chat body: one JSON object per line."""
    return io.BytesIO("".join(json.dumps(c) + "\n" for c in chunks).encode("utf-8"))


class FakeResponse(io.BytesIO):
    """urlopen's return value; also records the request it was given."""


class TestStreaming(unittest.TestCase):

    def fake_urlopen(self, body, sent):
        def urlopen(req, *a, **k):
            sent.append(json.loads(req.data))
            return FakeResponse(body.getvalue())
        return urlopen

    def test_read_stream_assembles_text_and_forwards_pieces(self):
        pieces = []
        reply = agent.read_stream(ndjson(
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ), pieces.append)
        self.assertEqual(pieces, ["Hel", "lo"])
        self.assertEqual(reply, {"role": "assistant", "content": "Hello"})

    def test_read_stream_collects_tool_calls_whole(self):
        call = {"function": {"name": "web_search", "arguments": {"query": "x"}}}
        reply = agent.read_stream(ndjson(
            {"message": {"content": "", "tool_calls": [call]}, "done": False},
            {"message": {"content": ""}, "done": True},
        ), lambda t: None)
        self.assertEqual(reply["tool_calls"], [call])

    def test_read_stream_raises_on_error_chunk(self):
        with self.assertRaises(RuntimeError):
            agent.read_stream(ndjson({"error": "model not found"}), lambda t: None)

    def test_without_on_token_the_request_is_not_streamed(self):
        """The terminal agent keeps v7's non-streaming request."""
        sent = []
        body = io.BytesIO(json.dumps({"message": {"content": "hi"}}).encode())
        with mock.patch("urllib.request.urlopen", self.fake_urlopen(body, sent)):
            reply = agent.call_ollama("m", [], schemas=[])
        self.assertIs(sent[0]["stream"], False)
        self.assertEqual(reply["content"], "hi")

    def test_with_on_token_the_request_streams(self):
        sent, pieces = [], []
        body = ndjson({"message": {"content": "a"}, "done": False},
                      {"message": {"content": "b"}, "done": True})
        with mock.patch("urllib.request.urlopen", self.fake_urlopen(body, sent)):
            reply = agent.call_ollama("m", [], schemas=[], on_token=pieces.append)
        self.assertIs(sent[0]["stream"], True)
        self.assertEqual((pieces, reply["content"]), (["a", "b"], "ab"))

    def test_run_agent_resets_text_streamed_before_a_tool_call(self):
        """Text that preceded a tool call was not the answer: discard it."""
        events = []
        replies = iter([
            {"role": "assistant", "content": "Let me check.",
             "tool_calls": [{"function": {"name": "get_current_time", "arguments": {}}}]},
            {"role": "assistant", "content": "It is noon."},
        ])
        view = agent.build_view("general")
        with mock.patch.dict(view.registry, {"get_current_time": lambda: "NOON"}), \
                mock.patch.object(agent, "call_ollama",
                                  side_effect=lambda *a, **k: next(replies)), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            agent.run_agent("m", [{"role": "user", "content": "time?"}],
                            schemas=view.schemas, registry=view.registry,
                            on_token=events.append)
        self.assertEqual(events, [None])
        self.assertIn("agent> It is noon.", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
