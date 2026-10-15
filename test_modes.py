#!/usr/bin/env python3
"""
Tests for project modes (modes.py).

    python3 -m unittest test_modes -v

The property that matters most is a security one: the registry a mode hands
back must contain exactly the tools whose schemas it shows. If they diverge,
a model that guesses the name of a hidden tool gets it run anyway.
"""

import unittest

import modes
from tools import TOOL_REGISTRY, TOOL_SCHEMAS


def names(schemas):
    return [s["function"]["name"] for s in schemas]


class TestModes(unittest.TestCase):

    def view(self, mode):
        return modes.apply_mode(mode, "BASE", TOOL_SCHEMAS, TOOL_REGISTRY)

    def test_registry_matches_schemas_in_every_mode(self):
        for name in modes.MODES:
            v = self.view(name)
            self.assertEqual(set(v.registry), set(names(v.schemas)), name)

    def test_general_is_the_v6_tool_set(self):
        v = self.view("general")
        runnable = {n for n in names(TOOL_SCHEMAS) if callable(TOOL_REGISTRY.get(n))}
        self.assertEqual(set(names(v.schemas)), runnable)
        self.assertNotIn("get_prices", v.registry)
        self.assertEqual(v.prompt, "BASE")   # no skill in general mode

    def test_schema_without_implementation_is_hidden(self):
        """The fetch_url bug class: a schema the model can see but nothing
        can run. Hidden and reported, never shown."""
        reg = {k: v for k, v in TOOL_REGISTRY.items() if k != "web_search"}
        v = modes.apply_mode("general", "BASE", TOOL_SCHEMAS, reg)
        self.assertNotIn("web_search", names(v.schemas))
        self.assertIn("web_search (schema but no implementation)", v.missing)

    def test_finance_adds_market_and_drops_file_mutation(self):
        v = self.view("finance")
        n = names(v.schemas)
        self.assertIn("get_prices", n)
        self.assertIn("run_python", n)
        for hidden in ("edit_file", "delete_file"):
            self.assertNotIn(hidden, n)
            self.assertNotIn(hidden, v.registry)
        self.assertIs(v.registry["get_prices"].__module__, "market")

    def test_math3332_has_editing_but_no_market(self):
        n = names(self.view("math3332").schemas)
        self.assertIn("edit_file", n)
        self.assertNotIn("get_prices", n)
        self.assertNotIn("delete_file", n)

    def test_skill_is_appended_after_base_prompt(self):
        v = self.view("finance")
        self.assertTrue(v.prompt.startswith("BASE\n\n--- Mode: finance ---\n"))
        self.assertIn("No lookahead", v.prompt)

    def test_include_order_is_preserved(self):
        """Order is part of the prompt, so it's fixed per mode — a stable
        prefix is what lets Ollama cache it."""
        v = self.view("finance")
        wanted = [t for t in modes.MODES["finance"].include if t in names(v.schemas)]
        self.assertEqual(names(v.schemas), wanted)

    def test_missing_tools_are_reported_not_fatal(self):
        partial = [s for s in TOOL_SCHEMAS if s["function"]["name"] != "fetch_url"]
        v = modes.apply_mode("finance", "BASE", partial, TOOL_REGISTRY)
        self.assertIn("fetch_url", v.missing)
        self.assertNotIn("fetch_url", v.registry)

    def test_junk_registry_keys_are_scrubbed(self):
        """The v5 merge bug put a phantom 'type' key in the registry."""
        dirty = {**TOOL_REGISTRY, "type": "function"}
        v = modes.apply_mode("general", "BASE", TOOL_SCHEMAS, dirty)
        self.assertNotIn("type", v.registry)

    def test_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "choose from"):
            self.view("crypto")

    def test_every_mode_stays_within_the_tool_budget(self):
        for name in modes.MODES:
            self.assertLessEqual(len(self.view(name).schemas), 14, name)

    def test_without_skill_the_prompt_is_the_base_prompt(self):
        for name in modes.MODES:
            v = modes.apply_mode(name, "BASE", TOOL_SCHEMAS, TOOL_REGISTRY, with_skill=False)
            self.assertEqual(v.prompt, "BASE", name)
            # The tool subset is unchanged: only the skill text is dropped.
            self.assertEqual(names(v.schemas), names(self.view(name).schemas), name)

    def test_every_referenced_skill_file_exists(self):
        for mode in modes.MODES.values():
            if mode.skill:
                self.assertTrue((modes.SKILLS_DIR / mode.skill).exists(), mode.skill)
                self.assertTrue(modes.load_skill(mode.skill))


if __name__ == "__main__":
    unittest.main(verbosity=2)
