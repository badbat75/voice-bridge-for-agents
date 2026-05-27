# resources/ — deployment templates

Copy-and-fill templates for a fresh checkout. None of these are read directly by
the bridge; they are starting points you copy into place.

| Template | Copy to | Purpose |
| --- | --- | --- |
| `voice-bridge.secrets.example.json` | `../voice-bridge.secrets.json` | API keys + gateway token (gitignored once filled). |
| `asoundrc.example` | `~/.asoundrc` | ALSA `voice_out` PCM + `VoiceBridge` softvol control that `voice-bridge.json`'s `output_device` points at. |

```bash
cp resources/voice-bridge.secrets.example.json voice-bridge.secrets.json   # then fill in keys
cp resources/asoundrc.example ~/.asoundrc                                   # adjust card index if needed
```
