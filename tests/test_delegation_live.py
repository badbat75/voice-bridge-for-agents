#!/usr/bin/env python3
"""Live end-to-end: async delegation through the real zeroclaw, with timings.

Manual test, like `test_hid_interactive.py`: it talks to the real voice
agent in the bridge's own session, the worker really runs, and the outcome
is SPOKEN on the Jabra. Skips when the gateway or the bridge is down.

What it drives:

    test ──WS──▶ voice agent ──delegate(background=true)──▶ worker (sleep N)
      ◀─ "ci penso, ti avviso" ─┘                              │
                                        report_to_user ◀───────┘
    bridge ◀─ speaker socket ◀─ MCP sidecar
    bridge ──WS──▶ voice agent ──▶ TTS ──▶ speaker

and what it measures (seconds from the moment the message is sent):

    announce     first answer text from the voice agent
    delegate     the `delegate` tool call
    turn_done    the voice agent's turn is over   ← must come BEFORE the worker ends
    task_seen    the worker's first LLM request (zeroclaw runtime trace)
    worker_done  the worker's final response
    report_in    the bridge logs "Report from … queued"
    report_turn  the bridge hands it to the voice agent
    spoken       first audio of the voice agent's relay

Each stage has its own failure message, so a missing piece of zeroclaw
config (worker without the tool, agent delegating synchronously) is named.

Run:  .venv/bin/python tests/test_delegation_live.py
Env:  VB_LIVE_SLEEP (worker task length, default 15), VB_LIVE_TIMEOUT (default 240),
      VB_LIVE_PROMPT (say it the way a user would, to test the agents' own
      instructions instead of the explicit recipe)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import bridge_config  # noqa: E402
import gateway  # noqa: E402

_TRACE = os.path.expanduser("~/.zeroclaw/data/state/runtime-trace.jsonl")
_SLEEP = int(os.environ.get("VB_LIVE_SLEEP", "15"))
_TIMEOUT = int(os.environ.get("VB_LIVE_TIMEOUT", "240"))
_UNIT = "voice-bridge.service"

_PROMPT = (
    "Test tecnico del canale asincrono, niente domande. Delega al worker IN BACKGROUND "
    "(delegate con background true) questo compito: eseguire il comando shell "
    "`sleep {n}; date +%H:%M:%S` e, a lavoro finito, chiamare il tool report_to_user "
    "con l'orario ottenuto. Tu rispondi subito in una frase che l'hai avviato, senza "
    "aspettare il risultato."
)


def _alive(url: str) -> bool:
    try:
        urllib.request.urlopen(url, timeout=1.5)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _bridge_active() -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", _UNIT]).returncode == 0


def _journal_since(epoch: float) -> list[tuple[float, str]]:
    out = subprocess.run(
        ["journalctl", f"_SYSTEMD_USER_UNIT={_UNIT}", "--since", f"@{int(epoch)}",
         "--no-pager", "-o", "short-unix"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        stamp, _, rest = line.partition(" ")
        try:
            rows.append((float(stamp), rest))
        except ValueError:
            pass
    return rows


def _wait_journal(epoch: float, needle: str, timeout: float, after: float = 0.0):
    """Wall-clock time of the first journal line containing `needle`."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for stamp, rest in _journal_since(epoch):
            if needle in rest and stamp >= after:
                return stamp, rest
        time.sleep(1.0)
    return None, None


def _wait_trace(epoch: float, message: str, timeout: float, agent: str = "worker"):
    """(wall time, attributes) of the first trace event `message` by `agent`
    after `epoch`. The trace is a short rolling file, so poll it."""
    from datetime import datetime
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with open(_TRACE) as f:
                lines = f.read().splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("message") != message or (d.get("zeroclaw") or {}).get("agent_alias") != agent:
                continue
            stamp = datetime.fromisoformat(d["@timestamp"].replace("Z", "+00:00")).timestamp()
            if stamp >= epoch:
                return stamp, d.get("attributes") or {}
        time.sleep(1.0)
    return None


class LiveAsyncDelegation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = bridge_config.load_config()
        if cls.cfg.get("gateway_backend") != "zeroclaw_ws":
            raise unittest.SkipTest("gateway_backend is not zeroclaw_ws")
        if not _alive(cls.cfg["gateway_base_url"]):
            raise unittest.SkipTest("zeroclaw gateway not reachable")
        if not _bridge_active():
            raise unittest.SkipTest("voice-bridge is not running (nobody would take the report)")
        if not os.path.exists(cls.cfg["speaker_socket"]):
            raise unittest.SkipTest("speaker socket missing: set mcp_server.enabled")

    def test_departure_and_return_with_timings(self):
        cfg = self.cfg
        marks: dict[str, float] = {}
        tools: list[dict] = []
        wall0, t0 = time.time(), time.monotonic()

        def on_event(ftype, frame):
            if ftype == "tool_call":
                tools.append(frame)
                if "delegate" in json.dumps(frame):
                    marks.setdefault("delegate", time.monotonic() - t0)

        reply = []
        for delta in gateway.gateway_chat_stream_zeroclaw_ws(
                cfg["gateway_base_url"], cfg["gateway_token"],
                os.environ.get("VB_LIVE_PROMPT") or _PROMPT.format(n=_SLEEP),
                cfg.get("gateway_agent", "default"), cfg.get("session_key", "voice-bridge"),
                on_event=on_event):
            if delta != gateway.TOOL_BOUNDARY:
                marks.setdefault("announce", time.monotonic() - t0)
                reply.append(delta)
        marks["turn_done"] = time.monotonic() - t0
        print(f"\nvoice agent: {''.join(reply).strip()!r}")
        print("tool calls: " + json.dumps(tools, ensure_ascii=False)[:600])

        try:
            # ---- departure
            self.assertIn("delegate", marks,
                          "the voice agent answered without calling `delegate`")
            self.assertTrue(reply, "the voice agent said nothing")
            self.assertLess(
                marks["turn_done"], marks["delegate"] + _SLEEP,
                f"the turn lasted as long as the worker's task ({_SLEEP}s sleep): the "
                "delegation was BLOCKING. The voice agent must pass background=true.")

            # ---- the worker runs (followed in zeroclaw's runtime trace)
            seen = _wait_trace(wall0, "llm_request", 30)
            self.assertIsNotNone(seen, "the delegate call returned but no worker run "
                                       f"showed up in {_TRACE}")
            marks["task_seen"] = seen[0] - wall0
            done = _wait_trace(wall0, "turn_final_response", _TIMEOUT)
            self.assertIsNotNone(done, f"the worker did not finish within {_TIMEOUT}s")
            marks["worker_done"] = done[0] - wall0
            worker_said = str(done[1].get("text", ""))
            print(f"worker: {worker_said[:300]!r}")
            status = {"output": worker_said}

            # ---- return
            stamp, _ = _wait_journal(wall0, "Report from worker queued", 30)
            self.assertIsNotNone(
                stamp, "the worker finished but no report reached the bridge: is "
                "`voice_bridge` in [mcp_bundles.worker] and `voice_bridge__report_to_user` "
                "out of the worker's excluded_tools? Worker said: "
                f"{str(status.get('output'))[:200]!r}")
            marks["report_in"] = stamp - wall0
            stamp, _ = _wait_journal(wall0, "voice agent:", 60, after=stamp)
            self.assertIsNotNone(stamp, "the bridge queued the report but never delivered it "
                                        "(someone talking, or a reply still playing?)")
            marks["report_turn"] = stamp - wall0
            stamp2, line = _wait_journal(wall0, "Turn: first audio", 90, after=stamp)
            self.assertIsNotNone(stamp2, "the voice agent got the report but nothing was spoken")
            marks["spoken"] = stamp2 - wall0
            _s, said = _wait_journal(wall0, "INFO Binary:", 30, after=stamp)
            print(f"relay: {(said or '').split('INFO Binary:')[-1].strip()[:200]!r}")
        finally:
            self._table(marks)

    @staticmethod
    def _table(marks: dict) -> None:
        order = ["announce", "delegate", "turn_done", "task_seen", "worker_done",
                 "report_in", "report_turn", "spoken"]
        print("\n  stage         at (s)   step (s)")
        prev = 0.0
        for k in sorted(order, key=lambda k: marks.get(k, 1e9)):
            if k in marks:
                print(f"  {k:<12} {marks[k]:>7.1f}   {marks[k] - prev:>7.1f}")
                prev = marks[k]
        if {"worker_done", "spoken"} <= marks.keys():
            print(f"\n  user waits for the first answer : {marks.get('announce', 0):.1f} s")
            print(f"  worker's own run                : "
                  f"{marks['worker_done'] - marks.get('task_seen', 0):.1f} s (task sleeps {_SLEEP})")
            print(f"  worker done → spoken            : {marks['spoken'] - marks['worker_done']:.1f} s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
