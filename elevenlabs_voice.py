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
import wave

import elevenlabs

log = logging.getLogger(__name__)


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
        stt_model: str = "scribe_v1",
        stt_language: str = "ita",
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._tts_model = tts_model
        # ElevenLabs supports a fixed set of PCM rates: 8000, 16000,
        # 22050, 24000, 32000, 44100, 48000. Anything else → SDK error.
        self._tts_output_format = f"pcm_{int(tts_sample_rate)}"
        self._tts_language = tts_language
        self._tts_voice_settings = tts_voice_settings
        self._tts_text_normalization = tts_text_normalization
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
