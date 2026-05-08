#!/usr/bin/env python3
"""Unit tests for the deezer-connect ducking plugin.

Spins up a tiny in-process HTTP server that mimics the deezer-connect
BFF endpoints `GET /api/player/status` and `POST /api/player/volume`,
points the plugin at it, and asserts the duck/unduck round-trip.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deezer_connect_plugin import DeezerConnectPlugin


class _FakeBff:
    """Minimal stand-in for the deezer-connect BFF.

    Exposes a thread-safe `volume` attribute and a list of `requests`
    so tests can assert exactly which endpoints were hit and in what
    order.
    """

    def __init__(self, volume: int = 80) -> None:
        self.volume = volume
        self.requests: list[tuple[str, str, dict | None]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        bff = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args, **_kwargs):
                pass  # silence

            def _read_json(self):
                length = int(self.headers.get("Content-Length", 0))
                if not length:
                    return None
                return json.loads(self.rfile.read(length))

            def do_GET(self):  # noqa: N802
                if self.path == "/api/player/status":
                    with bff._lock:
                        bff.requests.append(("GET", self.path, None))
                        body = json.dumps({"volume": bff.volume}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def do_POST(self):  # noqa: N802
                if self.path == "/api/player/volume":
                    body = self._read_json() or {}
                    with bff._lock:
                        bff.requests.append(("POST", self.path, body))
                        if "volume" in body:
                            bff.volume = int(body["volume"])
                    self.send_response(200)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class DisabledPluginTests(unittest.TestCase):
    def test_disabled_by_default(self):
        # No config → plugin is off → duck/unduck must not touch HTTP.
        bff = _FakeBff(volume=80)
        url = bff.start()
        try:
            plug = DeezerConnectPlugin(None)
            plug.base_url = url  # would-be target if it weren't disabled
            plug.duck()
            plug.unduck()
            self.assertEqual(bff.requests, [])
            self.assertEqual(bff.volume, 80)
        finally:
            bff.stop()

    def test_explicit_disable(self):
        bff = _FakeBff(volume=80)
        url = bff.start()
        try:
            plug = DeezerConnectPlugin({"enabled": False, "base_url": url})
            plug.duck()
            plug.unduck()
            self.assertEqual(bff.requests, [])
        finally:
            bff.stop()


class EnabledPluginTests(unittest.TestCase):
    def _plugin(self, url: str, **overrides) -> DeezerConnectPlugin:
        cfg = {"enabled": True, "base_url": url, "ducking_percent": 50}
        cfg.update(overrides)
        return DeezerConnectPlugin(cfg)

    def test_duck_then_unduck_restores_volume(self):
        bff = _FakeBff(volume=80)
        url = bff.start()
        try:
            plug = self._plugin(url)
            plug.duck()
            self.assertEqual(bff.volume, 40)
            plug.unduck()
            self.assertEqual(bff.volume, 80)

            methods_paths = [(m, p) for m, p, _ in bff.requests]
            self.assertEqual(methods_paths, [
                ("GET", "/api/player/status"),
                ("POST", "/api/player/volume"),
                ("POST", "/api/player/volume"),
            ])
            self.assertEqual(bff.requests[1][2], {"volume": 40})
            self.assertEqual(bff.requests[2][2], {"volume": 80})
        finally:
            bff.stop()

    def test_duck_is_idempotent(self):
        # Two duck() calls without an intervening unduck must NOT
        # re-read and overwrite the saved original (otherwise the
        # second duck would save the already-ducked value as
        # "original" and unduck would restore the wrong number).
        bff = _FakeBff(volume=80)
        url = bff.start()
        try:
            plug = self._plugin(url)
            plug.duck()
            plug.duck()
            self.assertEqual(bff.volume, 40)
            plug.unduck()
            self.assertEqual(bff.volume, 80)
        finally:
            bff.stop()

    def test_unduck_without_duck_is_noop(self):
        bff = _FakeBff(volume=80)
        url = bff.start()
        try:
            plug = self._plugin(url)
            plug.unduck()
            self.assertEqual(bff.requests, [])
            self.assertEqual(bff.volume, 80)
        finally:
            bff.stop()

    def test_custom_ducking_percent(self):
        bff = _FakeBff(volume=100)
        url = bff.start()
        try:
            plug = self._plugin(url, ducking_percent=25)
            plug.duck()
            self.assertEqual(bff.volume, 25)
            plug.unduck()
            self.assertEqual(bff.volume, 100)
        finally:
            bff.stop()

    def test_zero_volume_is_noop(self):
        # Already at 0 → ducked target equals current → skip the POST
        # entirely so we don't pollute the BFF's state.
        bff = _FakeBff(volume=0)
        url = bff.start()
        try:
            plug = self._plugin(url)
            plug.duck()
            self.assertEqual(bff.volume, 0)
            self.assertEqual(
                [(m, p) for m, p, _ in bff.requests],
                [("GET", "/api/player/status")],
            )
        finally:
            bff.stop()

    def test_unreachable_bff_is_swallowed(self):
        # Closed port → URLError → plugin logs and returns without
        # crashing. Picks an ephemeral port and immediately closes it
        # so the connection refuses cleanly.
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        plug = DeezerConnectPlugin({
            "enabled": True,
            "base_url": f"http://127.0.0.1:{port}",
            "request_timeout_s": 0.5,
        })
        plug.duck()
        plug.unduck()  # both should return cleanly


if __name__ == "__main__":
    unittest.main(verbosity=2)
