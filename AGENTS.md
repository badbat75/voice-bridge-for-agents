# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python service that turns a Jabra SPEAK 510 USB speakerphone into a voice client for the OpenClaw gateway. No wake word and no PTT — once started, the mic stays open and an RMS-based endpointer commits each utterance after a configurable pause. **The bridge boots in mute** (LED red, USB-Audio capture firmware-silenced via output report `0x03`) and waits for the first Jabra HID button press to begin recording. A press while recording is a **soft "commit-and-mute"**: the endpointer flushes any in-progress speech buffer immediately (so the user's last utterance is still sent to STT/gateway/TTS and they hear the reply), the mic closes, and the device re-mutes — but the in-flight pipeline keeps flowing. A press while idle resumes recording. After `idle_timeout_ms` of pure silence the bridge auto-idles back to mute; if a turn was still being processed when that fires, the player un-idles itself when the reply starts so the user can talk back the moment it ends. The bridge writes the LED on every state transition, and that same write also flips the firmware capture mute — so the device's red ring and its actual mute state always agree with `recording`. It runs on a Raspberry Pi (ARM64, currently the only deployment target).

Pipeline (4 daemon threads + the HID monitor thread):

```
Recorder ──audio_q──▶ Endpointer ──utt_q──▶ Worker ──playback_q──▶ Player
```

Per utterance: PyAudio chunks flow continuously into `audio_q`; once the endpointer sees `silence_timeout_ms` of silence after speech, it pushes the buffered PCM to `utterance_q`. The worker pulls FIFO and runs STT → gateway `/v1/chat/completions` SSE → TTS streaming, pushing PCM into `playback_q`. The player owns one `aplay` subprocess per utterance. After `idle_timeout_ms` of pure silence (no speech), the bridge auto-idles: the recording stream closes and any in-flight processing is allowed to drain naturally.

Source files:
- `voice-bridge.py` — entry point + `VoiceBridge` orchestrator (4 worker threads, generation-counter-based hard-cancel, soft auto-idle). Keeps the streaming helpers (`gateway_chat`, `gateway_chat_stream`, `play_audio_stream`, `_aplay_popen`, `play_beep`, `find_input_device`) intact for tests and ad-hoc reuse.
- `jabra_hid.py` — `HidMuteMonitor`: Jabra HID button watcher (background thread, edge-triggered, USB unplug-resilient). Voice-bridge only calls `start() / stop() / consume_unmute_event()` on it.
- `deepgram_voice.py` — `DeepgramVoice`: Deepgram STT (`nova-3` Italian) + TTS (Aura, English/Spanish only — don't use for Italian). `transcribe(pcm, rate)` and `synthesize(text)`.
- `elevenlabs_voice.py` — `ElevenLabsVoice`: ElevenLabs TTS (`eleven_multilingual_v2`, Italian) + STT (Scribe). Same `transcribe` / `synthesize` contract — providers are swappable. **TTS output is `pcm_22050`** (raw S16LE 22050 Hz mono) so it feeds straight into the configured `aplay` invocation; don't change either side without changing the other.
- `deezer_connect_plugin.py` — `DeezerConnectPlugin`: optional ducking integration with the deezer-connect BFF (HTTP on `localhost:8980`). Disabled-by-default; a no-op unless `deezer_connect.enabled: true` is set in `voice-bridge.json`. Wraps the player thread's aplay lifecycle: `duck()` is called right before aplay starts, `unduck()` in the player's `finally` and again on `bridge.stop()` to recover from SIGTERM mid-reply.
- `tests/test_hid_mute.py` — automated unit tests for the HID monitor (uses `os.pipe()` as a fake `/dev/hidraw`, no Jabra needed).
- `tests/test_hid_interactive.py` — interactive smoke test against the real Jabra (run on the Pi with the device plugged in).
- `tests/test_voice_providers.py` — interface/contract tests for the two voice providers (verifies same surface, safe fallbacks on bad input, defaults pinned).
- `tests/test_gateway_integration.py` — end-to-end gateway leg: drives `load_config()` + `gateway_chat()` against the live gateway, skips when unreachable.

## Run / develop

```bash
# Activate the local venv (Python 3.13, ARM-aarch64 wheels) and run directly:
.venv/bin/python voice-bridge.py
```

`.venv/` is gitignored (too large at ~68 MB). To bring it up on a fresh checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

There's no global Python tooling for `deepgram`, `elevenlabs`, `pyaudio` on the host Pi, so always install into the local `.venv`, never `pip install --user`.

## Config — two files, merged at startup

`load_config()` reads:

1. `voice-bridge.json` (this directory): everything local to the bridge. Runtime knobs:
   - **Audio I/O**: `sample_rate` (mic), `chunk_size` (PyAudio frames per read), `tts_sample_rate` (TTS output + `aplay` rate — MUST match), `output_device` (ALSA PCM name).
   - **Endpointer / VAD**: `vad_rms_threshold` (per-chunk energy above which a chunk counts as speech; same `sum(s²)/sqrt(N)` metric the legacy code used, NOT true RMS — see *Architecture notes*), `silence_timeout_ms` (pause after speech that commits an utterance; values <600 ms split natural between-word pauses), `idle_timeout_ms` (cumulative silence after which the mic stream is closed; set to `0` to disable auto-idle).
   - **Control**: `hid_mute_enabled` (whether the Jabra HID button toggles the bridge), `stt_provider` / `tts_provider` (`elevenlabs` or `deepgram`), `tts_streaming_mode` (see *Architecture notes*), `session_key` (sent verbatim as `X-OpenClaw-Session-Key`; see session-key note below).

   Edit and `systemctl --user restart openclaw-voicebridge` to apply.
2. `~/.openclaw/openclaw.json` (gateway config, owned by the wider OpenClaw system): all secrets and voice/agent settings. Specifically pulls:
   - `gateway.auth.token` → bearer for `/v1/chat/completions`
   - `stt.deepgram.apiKey` (or `DEEPGRAM_API_KEY` env override) — only validated if a Deepgram provider is selected
   - `messages.tts.providers.elevenlabs.*` — the **whole** ElevenLabs TTS block is consumed: `apiKey`, `voiceId`, `modelId` (note camelCase — earlier code looked for `model` and silently fell back to a hardcoded multilingual_v2), `voiceSettings`, `languageCode`, `applyTextNormalization`. The bridge does not fabricate any of these values. The `voiceSettings` keys are converted from openclaw's camelCase (`similarityBoost`, `useSpeakerBoost`) to the SDK's snake_case (`similarity_boost`, `use_speaker_boost`) at load-config time.

> **Schema gotcha (gateway v2026.5.5).** Earlier revisions of this doc described `agents.defaults.voiceBridgeModel` and a top-level `voice.sessionKey` block in `~/.openclaw/openclaw.json`. **Both are now rejected by the gateway's config schema** — adding them makes the gateway hard-crash at startup with `Unrecognized key: "voiceBridgeModel"` / `Unrecognized key: "voice"` (at runtime it merely "skips reload", which masks the bomb until the next restart). The bridge still reads them defensively as a fallback, but in practice: model defaults to `"openclaw"` (→ gateway's default agent, currently `main`), and the session key lives in `voice-bridge.json`. Don't put either back into the gateway config without verifying the schema accepts them again.

When changing voice/agent behavior, edit `~/.openclaw/openclaw.json`, **not** this script — except for `session_key`, which lives in `voice-bridge.json` for the reason above. Don't hardcode keys in `voice-bridge.json`. To switch which voice provider is used, edit `voice-bridge.json`.

## Architecture notes that aren't obvious from one file

- **Always-on mic, four async stages.** `VoiceBridge` runs five daemon threads (HID poller + recorder + endpointer + worker + player) connected by `queue.Queue` instances. The recorder keeps PyAudio open while `recording.is_set()`; the endpointer commits utterances on `silence_timeout_ms` pauses; the worker drives the streaming STT/gateway/TTS pipeline; the player owns one `aplay` per utterance. CPU is low (recorder blocks in `stream.read`, endpointer is `O(chunk_size)` per ~64 ms frame) but no longer ~0% the way the old wake-on-press model was — that's an intentional trade for sub-second time-to-speak.
- **No hard cancel — every "stop" is soft.** HID press while recording sets `_force_commit` (the endpointer flushes any in-progress speech buffer at the next loop tick, instead of waiting for `silence_timeout_ms`) and clears `recording`. Queues are NOT drained, `_gen` is NOT bumped, the active aplay is NOT killed — the worker still picks the committed utterance off `utterance_q`, the player still plays the reply, the user just stops being listened to until the next press. Auto-idle (silence ≥ `idle_timeout_ms`) does the same thing minus the force-commit: clears `recording`, drains `audio_q` (the chunks the recorder pushed during the silence window are no longer interesting), sets `_auto_idled`. Resuming via HID press bumps `_gen` (sheds any half-built endpointer state under the old gen) and clears `_auto_idled`.
- **Auto-resume on playback** (`_auto_idled` flag). If the bridge auto-idled while a turn was still being processed, the player un-idles itself when the first PCM chunk arrives — sets `recording`, writes `set_led(muted=False)`, resets `silence_count` via `_idle_reset_pending`, and clears `_auto_idled`. No `_gen` bump (we're inside the same turn). The user can talk back the instant the reply ends instead of having to press HID first. Crucially, this only fires on *auto*-idle: if the user explicitly pressed HID to mute, `_auto_idled` is clear and the player respects the press — playback happens but the mic stays muted afterwards.
- **VAD metric is `sum(s²) / sqrt(N)`, NOT true RMS.** The legacy `record_until_silence` had operator-precedence accidents (`/ len(samples) ** 0.5` evaluates `**` first), and the threshold defaults were calibrated against this dimensionally-weird quantity. The new endpointer keeps the same formula on purpose so prior tunings of `vad_rms_threshold` carry over. If you ever fix it to true RMS, also re-tune the default by ~`sqrt(chunk_size)`.
- **HID monitor lives in `jabra_hid.py`** as `HidMuteMonitor`, runs in a daemon thread (`_poll_loop`) reading raw reports from `/dev/hidraw*`. Device discovery matches USB vendor `0B0E` / product `0422` via `/sys/class/hidraw/*/device/uevent`. Open is `O_RDWR` — required for the engage write. The poll loop owns the device lifecycle: if the Jabra is absent at startup or disappears mid-run, it parks in `_shutdown.wait(2.0)` (Event-based, kernel-blocked, ~0 CPU) and retries `_open_device()` every 2 s until it appears or `stop()` is called. `start()` no longer returns a success bool — there is no "device absent at startup" failure path; plug state is runtime state.
- **The Jabra needs an off-hook engage write** (`bytes([0x03, 0x01, 0x00])` = output report `0x03` with LED Off-Hook bit set) before it emits any telephony input reports. Without it, pressing mute is silent on every interface — hidraw, `/dev/input/event*`, ALSA. `start()` writes this once. Full protocol details and the diagnostic trail are in `docs/JABRA.md`.
- **Bit 4 of byte 1 of report `0x03` is the *Mic Mute button-press* state** (momentary: high while held, low when released), HID Telephony usage `0x2F`. A single press emits a 1→0 sequence; the monitor edge-triggers on the rising edge so each press = exactly one wake event. The bit is marked `Constant` in the descriptor, which is why the kernel input layer doesn't surface it as `KEY_MICMUTE`.
- **Set `JABRA_HID_DEBUG=1`** to log every incoming HID report's first 8 bytes in hex — useful when diagnosing a different device variant or unexpected report layout.
- **Set `VOICE_BRIDGE_DEBUG_AUDIO=1`** to surface aplay's stderr (drops `-q`, forwards lines to the Python logger via a daemon thread). Use it to spot `underrun!!!` lines when the streaming TTS pipeline can't keep aplay's ALSA buffer fed — otherwise underruns are silent except for an audible click.
- **Host-driven mute, not host-tracked.** The bridge is the source of truth for mute state. `HidMuteMonitor` exposes `consume_unmute_event()` (edge-triggered "user pressed") *and* `set_led(muted: bool)`, which writes output report `0x03` with the Off-Hook + Mute bits. **The Mute bit is also the firmware capture-mute switch on this device** (verified by PCM probe: 0/14400 vs ~14400/14400 non-zero samples per second), so a single write controls both the red LED ring and whether the USB-Audio endpoint emits silence. `VoiceBridge` keeps the `recording` Event and calls `set_led(muted=not recording.is_set())` on every transition (HID press, auto-idle), so the device's actual mute always matches the bridge's intent. The boot default is muted (`_muted_pref=True` in `HidMuteMonitor`); `_engage()` re-applies the preference on every reconnect, so an unplug/replug snaps the device back to whatever the bridge currently wants instead of resetting.
- **Session-key handling.** The bridge sends `X-OpenClaw-Session-Key: <value>` per turn. The gateway's `resolveSessionKey` accepts the header **verbatim** — it does NOT add the canonical `agent:<agentId>:` prefix. Anything that isn't already in `agent:<id>:<rest>` shape becomes an "orphaned" session that the UI hides until the gateway's next-boot migration canonicalizes it (one-shot, not per-request, and a no-op once the canonical key already exists). The pinned value `agent:main:voice-bridge` in `voice-bridge.json` keeps every turn flowing into the same canonical session under the `main` agent (the existing on-disk session is `agents/main/sessions/<sessionId>.jsonl` keyed `agent:main:voice-bridge`). Don't change to a non-canonical key unless you understand this.
- **Tests:**
  - `.venv/bin/python tests/test_hid_mute.py` — automated, no hardware required. Drives `HidMuteMonitor` over an `os.pipe()` and asserts the edge-detection logic.
  - `.venv/bin/python tests/test_hid_interactive.py [N]` — manual hands-on test on the Pi: prompts the operator to press the button N times (default 3) and checks that exactly N wake events fire with reasonable spacing and no spurious extras. Use this when porting to a different host, after udev/permissions changes, or to confirm a wire-protocol fix on real hardware.
  - `.venv/bin/python tests/test_voice_providers.py` — provider interface/contract tests, no network.
  - `.venv/bin/python tests/test_gateway_integration.py` — gateway leg integration test; auto-skips if the gateway isn't reachable.
- **deezer-connect ducking (`deezer_connect` block in `voice-bridge.json`).** When `enabled: true`, the player thread asks the deezer-connect BFF for its current volume via `GET /api/player/status`, multiplies it by `ducking_percent/100` (default 50), writes the result back via `POST /api/player/volume`, and restores the original on playback end. Failures (BFF down, timeout, malformed JSON) are logged and swallowed — voice-bridge keeps playing the reply, just without the side-effect. The plugin is **off by default**: missing block or `enabled: false` makes `DeezerConnectPlugin.duck/unduck` no-ops, so it's free to leave the code path wired in. `request_timeout_s` (default 1.0) caps how long the player will wait on the BFF before giving up — keep it tight, since this call sits on the critical path between gateway delta arriving and aplay opening.
- **TTS streaming strategy (`tts_streaming_mode` in `voice-bridge.json`).** Default `"http_sentence"`: gateway deltas are buffered to a sentence boundary (`[.!?…]` + whitespace), then `text_to_speech.stream()` (HTTP) is called once per sentence and its PCM chunks are forwarded to `aplay`. Available on every ElevenLabs tier; first-audio latency ≈ first sentence. Alternative `"websocket"`: feeds deltas straight into `text_to_speech.convert_realtime` for token-level latency, but **requires a paid ElevenLabs tier** — free accounts get HTTP 403 on the WS upgrade. Only consulted when `tts_provider == "elevenlabs"`. The `language_code` and `apply_text_normalization` settings are forwarded only on the HTTP path; the WS endpoint rejects them, so don't rely on either when running in `websocket` mode.
- **Italian is hardcoded** in `transcribe_audio` (`language="it"`) and the fallback error string. If multilingual support is needed, plumb it through config.
- **TTS / gateway calls use `urllib.request`**, not `httpx`/the SDKs — keeps dependency surface minimal on ARM. Deepgram is the only one using its async SDK.
- **Output device** is an ALSA PCM name (default `plug:jabra_dmix` from `~/.asoundrc`), not a card index. `aplay` is invoked as a subprocess with `S16_LE @ 22050 Hz mono` — that sample rate is fixed to ElevenLabs' default output, so don't change it without also changing the TTS request.
- **`find_input_device`** matches "jabra" case-insensitively in PyAudio's device list; there is no equivalent fallback for output (output goes through ALSA by name).

## Deployment context

Run as a long-lived **user** systemd process on the Pi. `SIGTERM`/`SIGINT` are handled — it finishes any in-flight turn, stops the HID thread, and exits cleanly. Don't add a PID file or daemonization here; that's the unit's job.

`systemd/openclaw-voicebridge.service` is a user unit (linked into `~/.config/systemd/user/`, no `User=` directive). It runs as the invoking user (`openclaw`) and inherits that user's groups; `audio` is needed for ALSA, `plugdev` for /dev/hidraw* (granted by the udev rule). Linger (`loginctl enable-linger`) makes the user-mode systemd start at boot without a login session.

`systemd/99-openclaw-voicebridge.rules` (udev) sets `GROUP=plugdev MODE=0660` on the Jabra's hidraw node so the user can open it. The rule does **not** trigger the service — the bridge's reconnect loop handles plug/unplug on its own (`_poll_loop` parks on `_shutdown.wait(2.0)` between `_find_device()` attempts). This decoupling is what enabled moving from system to user systemd: udev can't directly start user units.

`systemd/install.sh` does the link + reload + linger + group sanity checks; see `systemd/README.md` for details. The `openclaw-` prefix matches the Pi's passwordless sudo policy for this user, though most operations on the user unit don't need sudo at all.
