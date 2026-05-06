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
import time
import unittest
import urllib.error
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_voice_bridge():
    """`voice-bridge.py` has a hyphen in its name, so a normal `import`
    won't work — load it via importlib so we can call its real
    `load_config` and `gateway_chat`."""
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_HERE, "voice-bridge.py"),
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

    def test_chat_completion_round_trip(self) -> None:
        """Send a short Italian prompt, expect a non-empty reply that
        is NOT the bridge's hardcoded exception fallback."""
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
            "Mi dispiace, ho avuto un problema di connessione.",
            reply,
            "gateway_chat hit its exception fallback — request failed",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
