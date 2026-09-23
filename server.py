#!/usr/bin/env python3
"""
v6 — A very small web UI for the agent.

    python3 server.py        then open http://localhost:8765

WHY IT LOOKS LIKE THIS

The agent's approval prompts (run_python, write_file, edit_file, delete_file)
call input(). There is no terminal in a browser, so a naive web port either
hangs forever or — far worse — drops the approval step and quietly removes the
only control standing between a steerable 8B model and your filesystem.

So approvals stay in the terminal, and the server just moves your attention:
when a prompt is pending it raises Terminal.app, and when the turn finishes it
raises the browser again. That is cosmetic AppleScript ("activate" only, no
scripting of page content); macOS asks permission the first time and the app
works fine if you refuse.

TWO THINGS THAT ARE LOAD-BEARING

1. 127.0.0.1 only. This agent reads files and runs code. Bound to 0.0.0.0 it
   would hand that to anyone on the same wifi. Never change HOST.
2. ThreadingHTTPServer. The chat request blocks on input() for as long as you
   take to answer, so /status must be served on another thread or the page can
   never show "waiting for approval" — it would just look frozen.

STATE is a single conversation, because this is a personal tool on localhost.
Two browser tabs would share and corrupt one history; the lock makes the second
tab wait rather than interleave.
"""

import http.server
import json
import platform
import subprocess
import sys
import threading
import webbrowser

import agent
import files
from memory import load_memory
from tools import TOOL_REGISTRY, TOOL_SCHEMAS

HOST, PORT = "127.0.0.1", 8765
IS_MAC = platform.system() == "Darwin"

STATE = {
    "messages": [],
    "status": "idle",     # idle | thinking | approval | error
    "detail": "",
    "turn": 0,            # bumped when a turn ends, so the page knows to refetch
}
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# Window focus
# --------------------------------------------------------------------------

def _activate(app: str) -> None:
    """Raise an app's window. Cosmetic only — no scripting of its contents.

    Silently does nothing if the user denied Automation permission, which is
    the correct failure mode: you alt-tab yourself and everything still works.
    """
    if not IS_MAC:
        return
    try:
        subprocess.run(["osascript", "-e", f'tell application "{app}" to activate'],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def _to_terminal() -> None:
    _activate("Terminal")


def _to_browser() -> None:
    if IS_MAC:
        # `open` raises whichever browser owns the tab, so this works for
        # Chrome, Safari, Arc or anything else without hardcoding a name.
        try:
            subprocess.run(["open", f"http://{HOST}:{PORT}"],
                           capture_output=True, timeout=5)
        except Exception:
            pass


# --------------------------------------------------------------------------
# Approval interception
# --------------------------------------------------------------------------

def _wrap_approvals() -> None:
    """Make every approval prompt raise Terminal first, and the browser after.

    The tools keep their own input() prompt untouched — this only wraps them,
    so the terminal interaction you already tested is exactly what runs.
    """
    def wrap(mod, name):
        original = getattr(mod, name)

        def wrapped(*a, **kw):
            with LOCK:
                STATE["status"] = "approval"
                STATE["detail"] = "Waiting for your approval in Terminal…"
            _to_terminal()
            try:
                return original(*a, **kw)
            finally:
                with LOCK:
                    STATE["status"] = "thinking"
                    STATE["detail"] = ""
                _to_browser()

        setattr(mod, name, wrapped)

    import sandbox
    wrap(sandbox, "_ask_approval")
    wrap(files, "_ask")


# --------------------------------------------------------------------------
# Running a turn
# --------------------------------------------------------------------------

def _run_turn(text: str, opts: dict) -> str:
    """Run one agent turn, capturing what it would have printed.

    run_agent prints the final answer rather than returning it, so stdout is
    redirected into a buffer. The terminal still sees the approval prompts,
    because those are written by input() to the real stdout before this swap
    takes effect inside the tool call... which is why the prompt text is
    printed to sys.__stdout__ in the tools, not through this buffer.
    """
    import io
    import contextlib

    STATE["messages"].append({"role": "user", "content": text})
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            agent.run_agent(
                opts["model"], STATE["messages"],
                verbose=True, num_ctx=opts["ctx"],
                think=opts["think"], show_time=False,
            )
    except Exception as exc:
        return f"[error] {exc}"
    return buffer.getvalue()


def _split_output(raw: str) -> tuple[str, list[str]]:
    """Separate the final answer from the tool trace."""
    answer, trace = "", []
    for line in raw.splitlines():
        if line.startswith("agent> "):
            answer = line[len("agent> "):]
        elif line.strip():
            trace.append(line.rstrip())
    return answer or "(no answer)", trace


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    opts: dict = {}

    def log_message(self, *args):
        pass  # the terminal belongs to the approval prompts

    def _send(self, code, body, ctype="application/json"):
        payload = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/status":
            with LOCK:
                self._send(200, json.dumps({
                    "status": STATE["status"],
                    "detail": STATE["detail"],
                    "turn": STATE["turn"],
                    "messages": len(STATE["messages"]),
                }))
        elif self.path == "/context":
            report = agent.context_report(
                STATE["messages"][0]["content"] if STATE["messages"] else "",
                STATE["messages"], self.opts["ctx"])
            self._send(200, json.dumps({"text": report}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/chat":
            self._send(404, json.dumps({"error": "not found"}))
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            text = json.loads(self.rfile.read(length)).get("text", "").strip()
        except Exception:
            self._send(400, json.dumps({"error": "bad request"}))
            return
        if not text:
            self._send(400, json.dumps({"error": "empty message"}))
            return

        # One conversation, one turn at a time. A second tab waits here.
        with LOCK:
            busy = STATE["status"] in ("thinking", "approval")
        if busy:
            self._send(409, json.dumps({"error": "a turn is already running"}))
            return

        with LOCK:
            STATE["status"] = "thinking"
            STATE["detail"] = ""

        if text == "/clear":
            with LOCK:
                STATE["messages"] = [{"role": "system",
                                      "content": agent.build_system_prompt()}]
                STATE["status"] = "idle"
                STATE["turn"] += 1
            self._send(200, json.dumps({"answer": "History cleared, memory reloaded.",
                                        "trace": []}))
            return

        if text == "/undo":
            result = files.undo_last()
            with LOCK:
                STATE["status"] = "idle"
            self._send(200, json.dumps({"answer": result, "trace": []}))
            return

        raw = _run_turn(text, self.opts)
        answer, trace = _split_output(raw)

        with LOCK:
            STATE["status"] = "idle"
            STATE["detail"] = ""
            STATE["turn"] += 1
        self._send(200, json.dumps({"answer": answer, "trace": trace}))


PAGE = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>local agent</title>
<style>
  :root { --bg:#faf9f7; --fg:#1c1b19; --dim:#6b6862; --line:#e2dfd9;
          --user:#eceae5; --warn:#8a5a00; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#171614; --fg:#e8e6e1; --dim:#8f8b83; --line:#2e2c28;
            --user:#232220; --warn:#d8a441; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6
         ui-sans-serif, -apple-system, system-ui, sans-serif;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           font-size:13px; color:var(--dim); display:flex; gap:14px; }
  header b { color:var(--fg); font-weight:600; }
  #log { flex:1; overflow-y:auto; padding:20px 18px; max-width:760px;
         width:100%; margin:0 auto; }
  .msg { margin-bottom:20px; white-space:pre-wrap; word-wrap:break-word; }
  .msg.you { background:var(--user); padding:10px 14px; border-radius:10px; }
  .who { font-size:12px; color:var(--dim); margin-bottom:4px; }
  details { margin-top:8px; font-size:13px; color:var(--dim); }
  details pre { white-space:pre-wrap; margin:6px 0 0; font:12px/1.5
                ui-monospace, SFMono-Regular, Menlo, monospace; }
  #welcome h1 { font-size:22px; font-weight:600; margin:10px 0 4px; }
  #welcome > p { color:var(--dim); margin:0 0 18px; }
  .start { display:flex; align-items:center; gap:12px; width:100%; text-align:left;
           margin-bottom:8px; padding:14px; border:1px solid var(--line);
           border-radius:10px; background:transparent; color:var(--fg);
           font:inherit; cursor:pointer; }
  .start:hover { border-color:var(--dim); }
  .start .n { color:var(--dim); font-size:13px; min-width:14px; }
  .start .t { flex:1; }
  .start .s { font-size:12px; color:var(--dim); }
  .hint { font-size:13px; color:var(--dim); margin-top:18px; }
  .hint code { font:12px ui-monospace, SFMono-Regular, Menlo, monospace;
               background:var(--user); padding:1px 5px; border-radius:4px; }
  #note { padding:8px 18px; font-size:13px; color:var(--warn); display:none;
          border-top:1px solid var(--line); }
  form { display:flex; gap:8px; padding:14px 18px; border-top:1px solid var(--line);
         max-width:760px; width:100%; margin:0 auto; }
  input { flex:1; padding:11px 14px; font:inherit; color:var(--fg);
          background:transparent; border:1px solid var(--line); border-radius:9px; }
  input:focus { outline:none; border-color:var(--dim); }
  button { padding:11px 18px; font:inherit; border:none; border-radius:9px;
           background:var(--fg); color:var(--bg); cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
</style>

<header><b>local agent</b><span id="model"></span><span id="state">idle</span></header>
<div id="log">
  <div id="welcome">
    <h1>Hi, welcome back Noky!</h1>
    <p>What do you want to start with?</p>
    <button class="start" data-prompt="Let's work on the finance trading project. Remind me where we left off, then help me plan the next step.">
      <span class="n">1</span>
      <span class="t">Finance trading</span>
      <span class="s">in progress</span>
    </button>
    <button class="start" data-prompt="Let's work on MATH 3332 teaching assistant prep. Remind me where we left off, then help me plan the next step.">
      <span class="n">2</span>
      <span class="t">MATH 3332 — Teaching Assistant</span>
      <span class="s">in progress</span>
    </button>
    <p class="hint">Or just type below. <code>/clear</code> resets, <code>/undo</code> rolls back the last file change.</p>
  </div>
</div>
<div id="note"></div>
<form id="form">
  <input id="input" placeholder="Ask something…  (/clear, /undo)" autocomplete="off" autofocus>
  <button id="send">Send</button>
</form>

<script>
const log = document.getElementById('log');
const note = document.getElementById('note');
const state = document.getElementById('state');
const input = document.getElementById('input');
const send = document.getElementById('send');

function add(who, text, trace) {
  const wrap = document.createElement('div');
  wrap.className = 'msg' + (who === 'you' ? ' you' : '');
  const label = document.createElement('div');
  label.className = 'who';
  label.textContent = who;
  wrap.appendChild(label);
  wrap.appendChild(document.createTextNode(text));
  if (trace && trace.length) {
    const d = document.createElement('details');
    const s = document.createElement('summary');
    s.textContent = trace.length + ' tool step(s)';
    const pre = document.createElement('pre');
    pre.textContent = trace.join('\\n');
    d.appendChild(s); d.appendChild(pre); wrap.appendChild(d);
  }
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return wrap;
}

// Poll separately from the chat request: that request is blocked for the whole
// turn (including however long an approval sits unanswered), so this is the
// only way the page can learn an approval is pending.
setInterval(async () => {
  try {
    const s = await (await fetch('/status')).json();
    state.textContent = s.status;
    if (s.status === 'approval') {
      note.style.display = 'block';
      note.textContent = '⌘ ' + s.detail;
    } else {
      note.style.display = 'none';
    }
  } catch (e) { state.textContent = 'server down'; }
}, 700);

// The starter buttons and the text box go through the same path, so a click
// behaves exactly as if you had typed the prompt yourself.
document.querySelectorAll('.start').forEach(b => {
  b.addEventListener('click', () => ask(b.dataset.prompt));
});

document.getElementById('form').addEventListener('submit', (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (text) { input.value = ''; ask(text); }
});

async function ask(text) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  add('you', text);
  send.disabled = true;
  const pending = add('agent', 'thinking…');
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    pending.remove();
    add('agent', data.answer || data.error || '(nothing)', data.trace);
  } catch (err) {
    pending.remove();
    add('agent', 'Request failed: ' + err.message);
  }
  send.disabled = false;
  input.focus();
}
</script>
</html>
"""


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="web UI for the local agent")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--ctx", type=int, default=agent.DEFAULT_NUM_CTX)
    p.add_argument("--think", action="store_true")
    p.add_argument("--plan", action="store_true",
                   help="plan mode: file writes/edits/deletes are refused")
    p.add_argument("--port", type=int, default=PORT)
    args = p.parse_args()

    files.PLAN_MODE = args.plan
    _wrap_approvals()

    STATE["messages"] = [{"role": "system", "content": agent.build_system_prompt()}]
    Handler.opts = {"model": args.model, "ctx": args.ctx, "think": args.think}

    url = f"http://{HOST}:{args.port}"
    print(f"Local agent web UI — {url}")
    print(f"  model: {args.model}   ctx: {args.ctx}"
          f"{'   [PLAN MODE]' if args.plan else ''}")
    print(f"  tools: {len(TOOL_REGISTRY)}   MEMORY.md: "
          f"{'loaded' if load_memory().strip() else 'empty'}")
    print("\nApproval prompts appear HERE, in this terminal.")
    print("This window will raise itself when the agent needs you.")
    print("Ctrl-C to stop.\n")

    server = http.server.ThreadingHTTPServer((HOST, args.port), Handler)
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()


if __name__ == "__main__":
    main()
