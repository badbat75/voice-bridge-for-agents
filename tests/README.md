# tests/

How each test relates to the moving parts of the bridge, and what it
needs to actually run.

| File | Hardware | Network | Notes |
|------|----------|---------|-------|
| `test_hid_mute.py` | no | no | Drives `HidMuteMonitor` over an `os.pipe()` fake `/dev/hidraw`. Pure logic. |
| `test_hid_interactive.py` | **Jabra plugged in** | no | Manual: prompts the operator to press the button N times and asserts edge counts. Run on the Pi. |
| `test_voice_providers.py` | no | no | Provider interface/contract, defaults, config plumbing, factory wiring. Uses dummy API keys; no outbound calls. |
| `test_streaming_pipeline.py` | no | loopback only | SSE parser + ElevenLabs / Deepgram `synthesize_stream` (SDK mocked) + `play_audio_stream` (Popen mocked) + a glue test that wires all three. |
| `test_gateway_integration.py` | no | live gateway | Real round-trip against the OpenClaw gateway using the local `voice-bridge.json` + `voice-bridge.secrets.json`. Auto-skips if the gateway isn't reachable. |

## Running

Always use the local venv — none of the deps are installed system-wide:

```bash
.venv/bin/python tests/test_hid_mute.py
.venv/bin/python tests/test_voice_providers.py
.venv/bin/python tests/test_streaming_pipeline.py
.venv/bin/python tests/test_gateway_integration.py
.venv/bin/python tests/test_hid_interactive.py [N]   # default N=3
```

There is no `pytest` config — every file has a `__main__` that invokes
`unittest.main`. Run them individually; CI / pre-flight is just looping
the four hardware-free files.

## What each one really catches

- **`test_hid_mute.py`** — regressions in the bit-4 / report-`0x03`
  edge-detection in `jabra_hid.py`. If the Jabra firmware ever changes
  the report layout this test won't help (use the interactive one);
  but as long as the reported byte sequence is the same, this catches
  state-machine bugs (double-fire, missed edges, post-unplug behavior).

- **`test_hid_interactive.py`** — the protocol contract with real
  hardware. Run after udev / permissions / kernel changes, after
  porting to a different host, or when chasing reports of "button
  doesn't wake". It's the only test that reads from `/dev/hidraw*`.

- **`test_voice_providers.py`** — pins the things that would silently
  break audio quality without raising:
  - ElevenLabs `output_format` is always `pcm_<rate>` (raw PCM, not
    MP3) — drift here = silent garbage at playback.
  - Deepgram defaults to Italian + `nova-3`.
  - `detect_language=true` in `providerOptions` overrides
    `stt_language`.
  - The provider factory returns the right class for each
    `stt_provider` / `tts_provider` value.
  - `load_config()` parses provider selection and rejects unknowns.

- **`test_streaming_pipeline.py`** — the three-stage streaming
  pipeline introduced for low-latency playback:
  - `gateway_chat_stream()` parses OpenAI SSE correctly: yields
    `delta.content` strings, drops keepalives / blank lines / empty
    deltas, stops at `[DONE]`, falls back to a fixed reply on HTTP
    error, and yields the **first delta before the last** (timing
    assertion against a 200 ms-paced loopback server).
  - `ElevenLabsVoice.synthesize_stream()` forwards
    `voice_id / model_id / output_format` and the text iterator into
    `convert_realtime`, promotes a dict `voice_settings` into the
    pydantic `VoiceSettings`, drops empty audio chunks, and swallows
    SDK errors.
  - `DeepgramVoice.synthesize_stream()` is the buffer-and-call shim
    (no real streaming) — pinned so a future "real Deepgram WS
    streaming" rewrite is opt-in, not accidental.
  - `play_audio_stream()` invokes `aplay` with the right argv,
    writes chunks to stdin in order, writes the **first chunk before
    later chunks are produced** (real streaming, not buffered),
    and survives `BrokenPipeError`.
  - End-to-end: real loopback gateway + fake TTS + fake aplay,
    verifying delta order is preserved and the first audio reaches
    aplay before the last gateway delta lands.

- **`test_gateway_integration.py`** — the live-gateway leg of a turn.
  Runs `gateway_chat()` (non-streaming) and `gateway_chat_stream()`
  (SSE) against the actual gateway with the production session key,
  checks the assembled reply isn't the bridge's hardcoded
  `Mi dispiace…` exception fallback, and prints TTFT for the
  streaming version so a regression in SSE buffering would show up
  as a one-delta / equal-to-full-latency run.

## What to add when

- New behavior in `jabra_hid.py` (e.g. another button bit, LED
  control) → extend `test_hid_mute.py` with a fresh fake-hidraw
  fixture; only fall back to `test_hid_interactive.py` if the
  behavior can't be exercised over the pipe.
- New voice provider, new config key, or change to provider
  defaults → `test_voice_providers.py`.
- New stage in the streaming pipeline (e.g. sentence chunker, barge-in
  abort), or a tweak to TTS chunking / aplay invocation →
  `test_streaming_pipeline.py`. Keep the timing assertions —
  "first chunk reaches the next stage before the last is produced"
  is what guards the streaming property.
- New gateway endpoint or auth header → `test_gateway_integration.py`.
  Skip-on-unreachable is the convention; don't fail the suite when
  the gateway is down.
