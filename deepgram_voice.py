"""Deepgram-based STT + TTS client.

Wraps the deepgram-sdk async surfaces behind synchronous methods that
match elevenlabs_voice.ElevenLabsVoice — both classes expose
`transcribe(pcm, sample_rate) -> str` and `synthesize(text) -> bytes`,
so callers can swap providers by changing one constructor call.

NOTE on TTS: Deepgram Aura currently ships only English (`*-en`) and
Spanish (`*-es`) voices — there is NO Italian voice. Don't use this for
TTS in an Italian flow; prefer ElevenLabs `eleven_multilingual_v2`.
STT works for Italian via `nova-3`, which is the default here.

Output of synthesize() is raw S16LE mono PCM at `tts_sample_rate` Hz —
ready to feed to `aplay -f S16_LE -r <rate> -c 1` directly, no
container parsing.
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from typing import Any, Coroutine, Iterable, Iterator

import deepgram

log = logging.getLogger(__name__)


def _run_sync(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run an async coroutine from sync code, even when an event loop is
    already running in the calling thread (e.g. inside FastMCP / asyncio
    servers). When there's no live loop we use plain `asyncio.run`;
    otherwise we hand the coroutine to a one-shot worker thread that owns
    its own loop, which keeps Deepgram's async client off the caller's
    loop entirely.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class DeepgramVoice:
    """Deepgram STT + TTS provider.

    The SDK is async-first; we expose sync methods by wrapping with
    `_run_sync`, which uses `asyncio.run` when no loop is live and
    falls back to a worker thread with its own loop when called from
    inside one (FastMCP / asyncio servers). Each call constructs a
    fresh client (cheap) — keeps the class stateless w.r.t. event
    loops.

    `stt_options` is a free-form dict of additional kwargs forwarded to
    `transcribe_file` — that's where openclaw-driven settings like
    `punctuate`, `smart_format`, `detect_language` flow through. If
    `detect_language=True` is in the options, the explicit
    `stt_language` is dropped so Deepgram can auto-detect.
    """

    def __init__(
        self,
        api_key: str,
        stt_model: str = "nova-3",
        stt_language: str | None = "it",
        stt_options: dict | None = None,
        tts_model: str = "aura-2-thalia-en",
        tts_sample_rate: int = 22050,
    ) -> None:
        self._api_key = api_key
        self._stt_model = stt_model
        self._stt_language = stt_language
        self._stt_options = stt_options or {}
        self._tts_model = tts_model
        self._tts_sample_rate = tts_sample_rate

    def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe S16LE mono PCM. Returns "" on empty input or error."""
        if not pcm:
            return ""
        try:
            return _run_sync(self._transcribe_async(pcm, sample_rate))
        except Exception as exc:
            log.error("STT error: %s", exc)
            return ""

    def synthesize(self, text: str) -> bytes | None:
        """Generate speech audio. Returns raw S16LE mono PCM bytes, or
        None on error."""
        if not text:
            return None
        try:
            return _run_sync(self._synthesize_async(text))
        except Exception as exc:
            log.error("TTS error: %s", exc)
            return None

    def synthesize_stream(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        """Buffer all text deltas, then call `synthesize()` and yield the
        result as a single chunk.

        This is a *compatibility shim*, not real streaming. Aura's REST
        endpoint doesn't take incremental text input, and the bridge's
        Italian deployment uses ElevenLabs anyway — so the priority here
        is just to keep `synthesize_stream` callable on either provider.
        First-audio latency under this provider is therefore the full
        gateway-stream + full TTS round-trip, same as the non-streaming
        path.
        """
        text = "".join(text_iter).strip()
        if not text:
            return
        audio = self.synthesize(text)
        if audio:
            yield audio

    # -----------------------------------------------------------------------

    async def _transcribe_async(self, pcm: bytes, sample_rate: int) -> str:
        # Wrap raw PCM in WAV so the API auto-detects encoding/rate from
        # the header instead of needing extra query params.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)

        # Build kwargs starting with our own defaults, then layer in the
        # openclaw-provided options. Anything in stt_options wins.
        kwargs: dict = {
            "request": buf.getvalue(),
            "model": self._stt_model,
            "smart_format": True,
        }
        kwargs.update(self._stt_options)
        # Only pin language if no explicit `language` is in the options
        # AND the options aren't asking the API to detect language.
        if (
            self._stt_language is not None
            and "language" not in kwargs
            and not kwargs.get("detect_language")
        ):
            kwargs["language"] = self._stt_language

        client = deepgram.AsyncDeepgramClient(api_key=self._api_key)
        response = await client.listen.v1.media.transcribe_file(**kwargs)
        try:
            return response.results.channels[0].alternatives[0].transcript.strip()
        except (AttributeError, IndexError):
            return ""

    async def _synthesize_async(self, text: str) -> bytes:
        client = deepgram.AsyncDeepgramClient(api_key=self._api_key)
        chunks: list[bytes] = []
        async for chunk in client.speak.v1.audio.generate(
            text=text,
            model=self._tts_model,
            encoding="linear16",
            sample_rate=self._tts_sample_rate,
        ):
            chunks.append(chunk)
        return b"".join(chunks)
