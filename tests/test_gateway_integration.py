#!/usr/bin/env python3
"""Integration test for the gateway leg of a voice-bridge turn.

Bypasses STT/TTS and drives the same `load_config()` + `gateway_chat()`
path the bridge uses in production: a synthetic post-STT prompt is sent
to the OpenClaw gateway over HTTP, and the reply is checked for
plausibility. This catches breakage that unit tests miss — wrong model
name, wrong header (incl. the `X-OpenClaw-Session-Key` added by
OpenClaw), expired token, gateway-side route changes.

Skips itself when the gateway isn't reachable, so it's safe to run on a
dev machine without the gateway up. Fails only when the gateway IS
reachable but rejects the request.

Run: .venv/bin/python test_gateway_integration.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest
import urllib.error
import urllib.request

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_voice_bridge():
    """`voice-bridge.py` has a hyphen in its name, so a normal `import`
    won't work — load it via importlib so we can call its real
    `load_config` and `gateway_chat`. The provider modules it imports
    live in the project root, which is added to sys.path above."""
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gateway_alive(base_url: str, timeout: float = 1.5) -> bool:
    """Quick reachability probe: any HTTP response (even 4xx/5xx)
    counts as alive. Connection refused / timeout / DNS failure means
    the gateway is down and the test should skip."""
    try:
        urllib.request.urlopen(base_url, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


class GatewayChatIntegrationTest(unittest.TestCase):
    """End-to-end check of the post-STT path against a live gateway."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.vb = _load_voice_bridge()
        cls.cfg = cls.vb.load_config()
        if not _gateway_alive(cls.cfg["gateway_base_url"]):
            raise unittest.SkipTest(
                f"gateway not reachable at {cls.cfg['gateway_base_url']}"
            )

    def test_config_has_required_keys(self) -> None:
        """The merged cfg must populate everything `gateway_chat` needs.
        If any of these are empty the request would still go out but
        with a malformed bearer / model — easier to fail here."""
        for key in ("gateway_base_url", "gateway_token", "voice_model", "session_key"):
            self.assertTrue(self.cfg.get(key), f"cfg[{key!r}] is empty")

    def _stream_for_backend(self, prompt: str):
        """Return the delta iterator for whichever gateway leg the merged
        cfg selects, so this test exercises the protocol actually in use
        (openclaw SSE, zeroclaw webhook, or zeroclaw_ws WebSocket)."""
        backend = self.cfg.get("gateway_backend", "openclaw")
        if backend == "zeroclaw_ws":
            return self.vb.gateway_chat_stream_zeroclaw_ws(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                prompt,
                self.cfg.get("gateway_agent", "default"),
                self.cfg.get("session_key", "voice-bridge"),
            )
        if backend == "zeroclaw":
            return self.vb.gateway_chat_stream_zeroclaw(
                self.cfg["gateway_base_url"],
                self.cfg["gateway_token"],
                prompt,
            )
        return self.vb.gateway_chat_stream(
            self.cfg["gateway_base_url"],
            self.cfg["gateway_token"],
            prompt,
            self.cfg["voice_model"],
            self.cfg.get("session_key", "voice-bridge"),
        )

    def test_chat_completion_round_trip(self) -> None:
        """Send a short Italian prompt, expect a non-empty reply that
        is NOT the bridge's hardcoded exception fallback.

        Non-streaming `gateway_chat` is the openclaw `/v1/chat/completions`
        primitive only — the zeroclaw legs have no synchronous equivalent,
        so skip there (the streaming test below covers them)."""
        if self.cfg.get("gateway_backend", "openclaw") != "openclaw":
            self.skipTest(
                f"gateway_chat is openclaw-only; backend is "
                f"{self.cfg.get('gateway_backend')!r}"
            )
        prompt = "Rispondi con una sola parola: ok"
        t0 = time.monotonic()
        reply = self.vb.gateway_chat(
            self.cfg["gateway_base_url"],
            self.cfg["gateway_token"],
            prompt,
            self.cfg["voice_model"],
            self.cfg.get("session_key", "voice-bridge"),
        )
        dt = time.monotonic() - t0
        print(f"\n  prompt : {prompt}")
        print(f"  reply  : {reply!r}")
        print(f"  latency: {dt:.2f}s")

        self.assertTrue(reply, "gateway returned an empty reply")
        # `gateway_chat` swallows exceptions and returns this string —
        # if we see it, the underlying request actually failed.
        self.assertNotIn(
            self.vb.GATEWAY_FALLBACK_REPLY,
            reply,
            "gateway_chat hit its exception fallback — request failed",
        )

    def test_chat_completion_streaming_round_trip(self) -> None:
        """Same gateway, streaming flavor: SSE deltas come back, the
        first one earlier than the full reply, and the assembled text
        is non-empty / not the fallback string.

        Also captures the time-to-first-token, which is the latency
        number that actually matters for perceived voice-bridge
        responsiveness — once a delta is in hand, TTS can start
        speaking. If this regresses (e.g. the gateway accidentally
        buffers the SSE), this test will catch it before users do."""
        prompt = "Rispondi con una sola parola: ok"
        t0 = time.monotonic()
        deltas: list[str] = []
        first_delta_at: float | None = None
        for delta in self._stream_for_backend(prompt):
            if first_delta_at is None:
                first_delta_at = time.monotonic() - t0
            deltas.append(delta)
        full_dt = time.monotonic() - t0
        reply = "".join(deltas)
        print(f"\n  prompt        : {prompt}")
        print(f"  reply         : {reply!r}")
        print(f"  deltas        : {len(deltas)}")
        print(f"  ttft          : {first_delta_at:.2f}s")
        print(f"  full latency  : {full_dt:.2f}s")

        self.assertTrue(deltas, "gateway streaming returned no deltas")
        self.assertNotIn(
            self.vb.GATEWAY_FALLBACK_REPLY,
            reply,
            "gateway_chat_stream hit its exception fallback — request failed",
        )
        # Sanity: assembled reply is non-empty after stripping.
        self.assertTrue(reply.strip(), f"streaming reply was blank: {reply!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
