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
import unittest
from unittest import mock

import agent


def tool_call(name, **args):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": args}}]}


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

    def test_general_mode_is_the_v6_tool_set(self):
        view = agent.build_view("general")
        self.assertEqual(view.schemas, agent.TOOL_SCHEMAS)
        self.assertEqual(set(view.registry), set(agent.TOOL_REGISTRY))

    def test_finance_mode_has_12_tools(self):
        self.assertEqual(len(agent.build_view("finance").schemas), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
