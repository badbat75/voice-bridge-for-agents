# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Python service that turns a Jabra SPEAK 510 USB speakerphone into a push-to-talk voice client for the OpenClaw gateway. The HID mute button is the wake trigger — no wake word, no always-on microphone. It runs on a Raspberry Pi (ARM64, currently the only deployment target).

Pipeline per turn: HID button press → beep → record-until-silence (PyAudio) → Deepgram REST STT (`nova-3`, `language=it`) → POST to gateway `/v1/chat/completions` → ElevenLabs REST TTS → `aplay` to ALSA.

Source files:
- `voice-bridge.py` — entry point: config loading, recording, gateway call, main loop. Delegates STT/TTS to provider classes and HID to `jabra_hid`.
- `jabra_hid.py` — `HidMuteMonitor`: Jabra HID button watcher (background thread, edge-triggered, USB unplug-resilient). Voice-bridge only calls `start() / stop() / consume_unmute_event()` on it.
- `deepgram_voice.py` — `DeepgramVoice`: Deepgram STT (`nova-3` Italian) + TTS (Aura, English/Spanish only — don't use for Italian). `transcribe(pcm, rate)` and `synthesize(text)`.
- `elevenlabs_voice.py` — `ElevenLabsVoice`: ElevenLabs TTS (`eleven_multilingual_v2`, Italian) + STT (Scribe). Same `transcribe` / `synthesize` contract — providers are swappable. **TTS output is `pcm_22050`** (raw S16LE 22050 Hz mono) so it feeds straight into the configured `aplay` invocation; don't change either side without changing the other.
- `test_hid_mute.py` — automated unit tests for the HID monitor (uses `os.pipe()` as a fake `/dev/hidraw`, no Jabra needed).
- `test_hid_interactive.py` — interactive smoke test against the real Jabra (run on the Pi with the device plugged in).
- `test_voice_providers.py` — interface/contract tests for the two voice providers (verifies same surface, safe fallbacks on bad input, defaults pinned).

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

1. `voice-bridge.json` (this directory): everything local to the bridge. Runtime knobs (gateway URL, mic `sample_rate`, chunk size, silence timeout, ALSA `output_device`, `hid_mute_enabled`) AND provider selection (`stt_provider`, `tts_provider` — values: `elevenlabs` or `deepgram`) AND output sample rate (`tts_sample_rate`, used by the TTS request and `aplay` together — they MUST match). Edit and `sudo systemctl restart openclaw-voicebridge` to apply.
2. `~/.openclaw/openclaw.json` (gateway config, owned by the wider OpenClaw system): all secrets and voice/agent settings. Specifically pulls:
   - `gateway.auth.token` → bearer for `/v1/chat/completions`
   - `stt.deepgram.apiKey` (or `DEEPGRAM_API_KEY` env override) — only validated if a Deepgram provider is selected
   - `messages.tts.providers.elevenlabs.*` — the **whole** ElevenLabs TTS block is consumed: `apiKey`, `voiceId`, `modelId` (note camelCase — earlier code looked for `model` and silently fell back to a hardcoded multilingual_v2), `voiceSettings`, `languageCode`, `applyTextNormalization`. The bridge does not fabricate any of these values. The `voiceSettings` keys are converted from openclaw's camelCase (`similarityBoost`, `useSpeakerBoost`) to the SDK's snake_case (`similarity_boost`, `use_speaker_boost`) at load-config time.
   - `agents.defaults.voiceBridgeModel` → model name sent to gateway (default `openclaw:voice` — currently set to `openclaw/main` as a workaround for a broken `openclaw/voice` agent gateway-side)
   - `voice.sessionKey` → session identifier

When changing voice/agent behavior, edit `~/.openclaw/openclaw.json`, **not** this script. Don't hardcode keys in `voice-bridge.json`. To switch which voice provider is used, edit `voice-bridge.conf`.

## Architecture notes that aren't obvious from one file

- **Idle CPU is ~0% by design.** The PyAudio stream is opened only after `HidMuteMonitor` reports an unmute transition and is torn down (`stream.close()` + `pa.terminate()`) the moment recording ends — before STT/LLM/TTS run. The main loop otherwise sleeps in 200 ms ticks. Don't refactor to a long-lived audio stream.
- **HID monitor lives in `jabra_hid.py`** as `HidMuteMonitor`, runs in a daemon thread (`_poll_loop`) reading raw reports from `/dev/hidraw*`. Device discovery matches USB vendor `0B0E` / product `0422` via `/sys/class/hidraw/*/device/uevent`. Open is `O_RDWR` — required for the engage write. Failures are non-fatal: the script logs and continues without HID (you'd then have no wake trigger).
- **The Jabra needs an off-hook engage write** (`bytes([0x03, 0x01, 0x00])` = output report `0x03` with LED Off-Hook bit set) before it emits any telephony input reports. Without it, pressing mute is silent on every interface — hidraw, `/dev/input/event*`, ALSA. `start()` writes this once. Full protocol details and the diagnostic trail are in `docs/JABRA.md`.
- **Bit 4 of byte 1 of report `0x03` is the *Mic Mute button-press* state** (momentary: high while held, low when released), HID Telephony usage `0x2F`. A single press emits a 1→0 sequence; the monitor edge-triggers on the rising edge so each press = exactly one wake event. The bit is marked `Constant` in the descriptor, which is why the kernel input layer doesn't surface it as `KEY_MICMUTE`.
- **Set `JABRA_HID_DEBUG=1`** to log every incoming HID report's first 8 bytes in hex — useful when diagnosing a different device variant or unexpected report layout.
- **No host-side mute tracking.** The monitor only exposes `consume_unmute_event()` (edge-triggered "user pressed the button"); it does not model the device's persistent mute LED state. The main loop idles until the first press, regardless of the speakerphone's actual mute LED.
- **Tests:**
  - `.venv/bin/python test_hid_mute.py` — automated, no hardware required. Drives `HidMuteMonitor` over an `os.pipe()` and asserts the edge-detection logic.
  - `.venv/bin/python test_hid_interactive.py [N]` — manual hands-on test on the Pi: prompts the operator to press the button N times (default 3) and checks that exactly N wake events fire with reasonable spacing and no spurious extras. Use this when porting to a different host, after udev/permissions changes, or to confirm a wire-protocol fix on real hardware.
- **Italian is hardcoded** in `transcribe_audio` (`language="it"`) and the fallback error string. If multilingual support is needed, plumb it through config.
- **TTS / gateway calls use `urllib.request`**, not `httpx`/the SDKs — keeps dependency surface minimal on ARM. Deepgram is the only one using its async SDK.
- **Output device** is an ALSA PCM name (default `plug:jabra_dmix` from `~/.asoundrc`), not a card index. `aplay` is invoked as a subprocess with `S16_LE @ 22050 Hz mono` — that sample rate is fixed to ElevenLabs' default output, so don't change it without also changing the TTS request.
- **`find_input_device`** matches "jabra" case-insensitively in PyAudio's device list; there is no equivalent fallback for output (output goes through ALSA by name).

## Deployment context

Run as a long-lived process under systemd on the Pi. `SIGTERM`/`SIGINT` are handled — it finishes any in-flight turn, stops the HID thread, and exits cleanly. Don't add a PID file or daemonization here; that's the unit's job.

`systemd/openclaw-voicebridge.service` plus `systemd/99-openclaw-voicebridge.rules` (udev) make the unit auto-start when the Jabra is plugged in — see `systemd/README.md` for the install commands (uses `systemctl link` for the unit). The `openclaw-` prefix matches the Pi's passwordless sudo policy for this user. The bridge's HID monitor handles unplug/replug internally via a reconnect loop with 2 s backoff, so the unit can stay running across plug cycles. After unplug `_poll_loop` enters `_reopen()` and waits in `_shutdown.wait(2.0)` until the device returns or `stop()` fires.
