"""Optional deezer-connect ducking plugin.

When enabled in `voice-bridge.json` under the `deezer_connect` block,
the bridge lowers the deezer-connect player's volume while a TTS reply
is playing and restores it when playback ends. Disabled (no-op) unless
`enabled: true` is set explicitly — same shape as other optional knobs
in this codebase.

Talks to the deezer-connect BFF over HTTP:

    GET  /api/player/status   -> { "volume": int, ... }
    POST /api/player/volume   <- { "volume": int }

Failures (BFF down, timeout, malformed JSON) are logged and swallowed —
voice-bridge keeps running, just without the ducking side-effect.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request

log = logging.getLogger("voice-bridge.deezer")


class DeezerConnectPlugin:
    """Ducks deezer-connect's volume during TTS playback.

    State machine is trivial: `_original_volume` is the volume we read
    before ducking, or `None` when not currently ducked. `duck()` is a
    no-op if we're already ducked; `unduck()` is a no-op if we're not.
    The single lock makes start/stop pairs safe under rapid back-to-back
    playback cycles (each player loop iteration is one duck/unduck).

    Because the player applies volume writes with latency, a `duck()`
    fired right after an `unduck()` (the auto-idle → next-reply churn when
    utterances are queued) can read back the still-ducked level and
    compound it (100 → 50 → 25). `_last_ducked_to` / `_last_original`
    guard that: if a duck reads back exactly the level it last ducked to,
    it re-arms the restore to the true baseline instead of ducking again.
    """

    def __init__(self, cfg: dict | None) -> None:
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.base_url = str(cfg.get("base_url", "http://localhost:8980")).rstrip("/")
        # Volume target as a percentage of the current volume. Default
        # 50 matches the spec; clamp to [0, 100] so a misconfig can't
        # drive a negative or runaway value.
        pct = float(cfg.get("ducking_percent", 50))
        self.ducking_factor = max(0.0, min(100.0, pct)) / 100.0
        # Tight timeout: the BFF is on localhost, anything slower than
        # this is a sign the BFF is hung — better to skip ducking than
        # to delay the start of aplay.
        self.timeout = float(cfg.get("request_timeout_s", 1.0))
        # Baseline volume the player is pinned to when the plugin is
        # enabled, applied once at bridge startup (see
        # `apply_default_volume`). Ducking then works off whatever the
        # player is at when a reply plays. Clamp to [0, 100]; default 75.
        self.default_volume = max(0, min(100, int(cfg.get("default_volume_percent", 75))))

        self._lock = threading.Lock()
        self._original_volume: int | None = None
        # Remember the last (true baseline, ducked-to) pair so a re-duck
        # can tell an already-ducked read apart from a genuine fresh
        # volume. The deezer player applies volume writes with latency, so
        # right after `unduck()` a `duck()` GET can still read back the old
        # ducked level; without this it would duck *that* (100→50→25) when
        # several utterances are queued. Survives the unduck (we only need
        # `_original_volume` cleared to re-arm; these stay).
        self._last_original: int | None = None
        self._last_ducked_to: int | None = None

    # -- startup --------------------------------------------------------
    def apply_default_volume(self) -> None:
        """Pin the deezer-connect player to `default_volume` at startup.

        Best-effort and no-op when disabled — mirrors how the bridge
        re-asserts its own output softvol level on boot. Any BFF failure
        is logged and swallowed (the GET/POST helper already warns)."""
        if not self.enabled:
            return
        if self._request("POST", "/api/player/volume", {"volume": self.default_volume}) is None:
            return
        log.info("deezer-connect: default volume set to %d%%", self.default_volume)

    # -- HTTP helper ----------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict | None:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                if not raw:
                    return {}
                return json.loads(raw)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
            log.warning("deezer-connect %s %s failed: %s", method, path, exc)
            return None

    # -- ducking API ----------------------------------------------------
    def duck(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._original_volume is not None:
                return
            status = self._request("GET", "/api/player/status")
            if not status:
                return
            try:
                vol = int(status.get("volume", 0))
            except (TypeError, ValueError):
                return
            # If we read back exactly the level we last ducked to, the
            # player is still (or already) ducked — either a stale read
            # right after `unduck()` (the device applies volume with
            # latency) or it never came back up. Ducking again would
            # compound it (100 → 50 → 25) across queued utterances. Don't
            # re-duck; just re-arm the restore to the true baseline so the
            # next `unduck()` puts the user back where they started.
            if self._last_ducked_to is not None and vol == self._last_ducked_to:
                self._original_volume = self._last_original
                log.info("deezer-connect: already ducked at %d, re-arm restore → %s",
                         vol, self._last_original)
                return
            ducked = max(0, min(100, int(round(vol * self.ducking_factor))))
            if ducked == vol:
                return
            if self._request("POST", "/api/player/volume", {"volume": ducked}) is None:
                return
            self._original_volume = vol
            self._last_original = vol
            self._last_ducked_to = ducked
            log.info("deezer-connect: ducked %d → %d", vol, ducked)

    def unduck(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._original_volume is None:
                return
            vol = self._original_volume
            self._original_volume = None
        if self._request("POST", "/api/player/volume", {"volume": vol}) is None:
            log.warning("deezer-connect: failed to restore volume to %d", vol)
            return
        log.info("deezer-connect: restored to %d", vol)
