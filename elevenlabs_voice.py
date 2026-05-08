"""ElevenLabs-based TTS + STT (Scribe) client.

Same interface as deepgram_voice.DeepgramVoice — `transcribe(pcm, rate)`
and `synthesize(text)` — so the two providers are swappable.

For an Italian voice bridge this is the recommended TTS provider:
`eleven_multilingual_v2` produces natural Italian speech, while
Deepgram Aura has no Italian voices. STT (Scribe) supports Italian
via `language_code="ita"`.

Output of synthesize() is raw S16LE mono PCM at the configured
`tts_sample_rate` (default 22050 Hz). Whoever consumes the bytes —
e.g. `aplay -f S16_LE -r <rate> -c 1` — must use the same rate.
ElevenLabs only ships a fixed set of PCM rates (8000, 16000, 22050,
24000, 32000, 44100, 48000); anything else causes an SDK error.
"""

from __future__ import annotations

import io
import logging
import re
import wave
from typing import Iterable, Iterator

import elevenlabs

log = logging.getLogger(__name__)


VALID_TTS_STREAM_MODES = ("http_sentence", "websocket")

# Sentence boundary used by the http_sentence streaming path: one or more
# `. ! ? …` followed by whitespace. Tuned for Italian assistant replies —
# we accept the rare false positive (e.g. "Sig. Rossi") in exchange for
# starting playback as soon as the first sentence is complete.
_SENT_END_RE = re.compile(r"[.!?…]+\s+")


class ElevenLabsVoice:
    """ElevenLabs TTS + STT provider.

    All TTS-side behavior (voice, model, voice_settings, language,
    text normalization) is settable at construction time so the bridge
    can pass whatever the openclaw gateway config specifies, instead
    of baking values in here.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        tts_model: str = "eleven_multilingual_v2",
        tts_sample_rate: int = 22050,
        tts_language: str | None = None,
        tts_voice_settings: dict | None = None,
        tts_text_normalization: str | None = None,
        tts_stream_mode: str = "http_sentence",
        stt_model: str = "scribe_v1",
        stt_language: str = "ita",
    ) -> None:
        if tts_stream_mode not in VALID_TTS_STREAM_MODES:
            raise ValueError(
                f"unknown tts_stream_mode {tts_stream_mode!r}; "
                f"valid: {VALID_TTS_STREAM_MODES}"
            )
        self._api_key = api_key
        self._voice_id = voice_id
        self._tts_model = tts_model
        # ElevenLabs supports a fixed set of PCM rates: 8000, 16000,
        # 22050, 24000, 32000, 44100, 48000. Anything else → SDK error.
        self._tts_output_format = f"pcm_{int(tts_sample_rate)}"
        self._tts_language = tts_language
        self._tts_voice_settings = tts_voice_settings
        self._tts_text_normalization = tts_text_normalization
        self._tts_stream_mode = tts_stream_mode
        self._stt_model = stt_model
        self._stt_language = stt_language

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe S16LE mono PCM via Scribe. Returns "" on empty input
        or error."""
        if not pcm:
            return ""
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm)
            client = elevenlabs.ElevenLabs(api_key=self._api_key)
            result = client.speech_to_text.convert(
                model_id=self._stt_model,
                file=("audio.wav", buf.getvalue(), "audio/wav"),
                language_code=self._stt_language,
            )
            return (getattr(result, "text", "") or "").strip()
        except Exception as exc:
            log.error("STT error: %s", exc)
            return ""

    def synthesize(self, text: str) -> bytes | None:
        """Generate speech audio. Returns bytes in `tts_output_format`
        (default `pcm_22050` = raw S16LE mono 22050 Hz), or None on error.

        Only forwards optional parameters (voice_settings, language,
        text_normalization) when they were explicitly configured —
        leaves the SDK's own defaults in place otherwise.
        """
        if not text:
            return None
        try:
            kwargs = {
                "voice_id": self._voice_id,
                "text": text,
                "model_id": self._tts_model,
                "output_format": self._tts_output_format,
            }
            if self._tts_voice_settings is not None:
                kwargs["voice_settings"] = self._tts_voice_settings
            if self._tts_language is not None:
                kwargs["language_code"] = self._tts_language
            if self._tts_text_normalization is not None:
                kwargs["apply_text_normalization"] = self._tts_text_normalization

            client = elevenlabs.ElevenLabs(api_key=self._api_key)
            chunks = client.text_to_speech.convert(**kwargs)
            return b"".join(chunks)
        except Exception as exc:
            log.error("TTS error: %s", exc)
            return None

    def synthesize_stream(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        """Stream TTS as PCM chunks while text deltas arrive.

        Two strategies, picked by `tts_stream_mode` at construction time:

        - `"http_sentence"` (default): buffer deltas until a sentence
          boundary, then call the HTTP streaming endpoint
          (`text_to_speech.stream`) once per sentence and forward its
          chunks. Works on every account tier; first-audio latency is
          one sentence rather than one token.

        - `"websocket"`: open the realtime websocket
          (`text_to_speech.convert_realtime`, endpoint
          `/v1/text-to-speech/{voice}/stream-input`) and feed deltas
          straight in for token-level latency. Requires an ElevenLabs
          paid tier — free accounts get HTTP 403 on the upgrade.

        Both paths emit the same `pcm_<rate>` output the rest of the
        bridge expects, and both swallow SDK / transport errors (logged,
        then stop yielding) so the main loop keeps running.
        """
        if self._tts_stream_mode == "websocket":
            yield from self._synthesize_stream_websocket(text_iter)
        else:
            yield from self._synthesize_stream_http_sentence(text_iter)

    def _synthesize_stream_websocket(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        """Realtime websocket path. See `synthesize_stream` for context.

        Caveats inherited from `convert_realtime` (SDK 2.45.0):
        - `language_code` and `apply_text_normalization` are NOT forwarded
          (the websocket endpoint doesn't accept them). For Italian via
          `eleven_multilingual_v2` this is fine — the model auto-detects.
        - Requires a paid tier; free accounts get HTTP 403 on connect.
        """
        try:
            kwargs = {
                "voice_id": self._voice_id,
                "text": iter(text_iter),
                "model_id": self._tts_model,
                "output_format": self._tts_output_format,
            }
            if self._tts_voice_settings is not None:
                # convert_realtime expects a VoiceSettings model (calls
                # `.dict()` on it); the rest of the codebase keeps
                # voice_settings as a plain dict, so promote it here.
                from elevenlabs.types.voice_settings import VoiceSettings
                kwargs["voice_settings"] = VoiceSettings(**self._tts_voice_settings)

            client = elevenlabs.ElevenLabs(api_key=self._api_key)
            for chunk in client.text_to_speech.convert_realtime(**kwargs):
                if chunk:
                    yield chunk
        except Exception as exc:
            log.error("TTS streaming error: %s", exc)

    def _synthesize_stream_http_sentence(
        self, text_iter: Iterable[str]
    ) -> Iterator[bytes]:
        """HTTP-streaming path: one `text_to_speech.stream` call per
        complete sentence, audio chunks forwarded as they arrive.

        Sentences are split on `[.!?…]+` followed by whitespace; the
        residual buffer is flushed as a final sentence when the upstream
        text iterator ends. Anything that crosses a delta boundary is
        accumulated until a boundary is found.

        Each sentence-level call forwards `language_code` and
        `apply_text_normalization` if they were configured (the HTTP
        endpoint accepts them, unlike the websocket path).
        """
        client = elevenlabs.ElevenLabs(api_key=self._api_key)
        buf = ""
        try:
            for delta in text_iter:
                if not delta:
                    continue
                buf += delta
                while True:
                    m = _SENT_END_RE.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.end()].strip()
                    buf = buf[m.end():]
                    if sentence:
                        yield from self._stream_sentence(client, sentence)
            tail = buf.strip()
            if tail:
                yield from self._stream_sentence(client, tail)
        except Exception as exc:
            log.error("TTS streaming error: %s", exc)

    def _stream_sentence(self, client, sentence: str) -> Iterator[bytes]:
        """Run a single HTTP TTS-stream call and yield bytes chunks.

        Errors are caught here too so one bad sentence doesn't kill the
        whole reply — we log and skip ahead to the next sentence.
        """
        kwargs = {
            "voice_id": self._voice_id,
            "text": sentence,
            "model_id": self._tts_model,
            "output_format": self._tts_output_format,
        }
        if self._tts_voice_settings is not None:
            kwargs["voice_settings"] = self._tts_voice_settings
        if self._tts_language is not None:
            kwargs["language_code"] = self._tts_language
        if self._tts_text_normalization is not None:
            kwargs["apply_text_normalization"] = self._tts_text_normalization
        try:
            for chunk in client.text_to_speech.stream(**kwargs):
                if isinstance(chunk, bytes) and chunk:
                    yield chunk
        except Exception as exc:
            log.error("TTS sentence error (%r): %s", sentence[:60], exc)
