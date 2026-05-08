#!/usr/bin/env python3
"""Tests for the streaming pipeline: gateway SSE → TTS websocket → aplay.

Three-stage pipeline being exercised, plus the end-to-end glue:

  1. `gateway_chat_stream(...)` — parses OpenAI-style SSE from the
     gateway and yields content deltas. Tested against a real loopback
     `http.server` so the urllib + line-iter path is genuinely covered.

  2. `ElevenLabsVoice.synthesize_stream(text_iter)` — wraps the SDK's
     `convert_realtime` (websocket TTS). The SDK is monkey-patched here:
     opening a real websocket from a unit test would require auth + the
     actual ElevenLabs service, so we verify what we DO own — that the
     wrapper forwards voice/model/format and the text iterator straight
     through, and that audio chunks come back out unchanged.

  3. `play_audio_stream(device, audio_iter, rate)` — pipes audio chunks
     through `aplay` as they arrive. We replace `subprocess.Popen` with
     a tape recorder so we can assert that:
       - aplay is invoked with the right args (`-f S16_LE -r <rate>` etc.)
       - chunks reach stdin in order
       - the FIRST chunk is written before LATER chunks are produced
         (i.e. it really streams; it doesn't accidentally collect into
         memory and flush at the end).

  4. End-to-end — a fake gateway (loopback HTTP) + a fake TTS provider
     (yields one PCM blob per text delta) + a fake `aplay` (captures
     stdin in real time) verifies the full chain produces audio
     incrementally, in delta order.

Run: .venv/bin/python tests/test_streaming_pipeline.py
"""

from __future__ import annotations

import http.server
import importlib.util
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from typing import Iterable, Iterator
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _load_voice_bridge():
    """`voice-bridge.py`'s hyphen blocks normal `import`; load via
    importlib so the tests can call the same `gateway_chat_stream` /
    `play_audio_stream` symbols the bridge runs in production."""
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fake gateway: a tiny SSE-emitting http.server we can point urllib at.
# ---------------------------------------------------------------------------
class _SSEHandler(http.server.BaseHTTPRequestHandler):
    """Replays a script of SSE chunks set on the server instance.

    The script is a list of byte blobs; each blob is written to the
    socket in order, with a `flush_pause` between them so the client's
    line-iter genuinely sees them as separate reads. That's how we can
    later assert the pipeline reacts to the first delta before the
    server has finished sending.
    """

    def log_message(self, *_args, **_kw):  # silence default access logs
        pass

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        # Drain the request body so the client doesn't see a reset.
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length:
            self.rfile.read(length)

        status, sse_script, flush_pause = self.server.script  # type: ignore[attr-defined]
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if status >= 400:
            return

        for blob in sse_script:
            self.wfile.write(blob)
            self.wfile.flush()
            if flush_pause:
                time.sleep(flush_pause)


def _start_sse_server(script, status=200, flush_pause=0.0):
    """Bring up a one-shot threaded HTTP server on localhost.

    `script` is the list of raw byte blobs to emit on the next POST.
    Returns `(base_url, shutdown_callable)`.
    """
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    srv.script = (status, script, flush_pause)  # type: ignore[attr-defined]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_port}"

    def stop():
        srv.shutdown()
        srv.server_close()
        t.join(timeout=2)

    return base, stop


# ---------------------------------------------------------------------------
# 1. gateway_chat_stream — real loopback HTTP, real urllib.
# ---------------------------------------------------------------------------
class GatewayChatStreamTest(unittest.TestCase):
    def setUp(self):
        self.vb = _load_voice_bridge()

    def _run(self, script, status=200, flush_pause=0.0):
        base, stop = _start_sse_server(script, status=status, flush_pause=flush_pause)
        try:
            return list(self.vb.gateway_chat_stream(
                base, "tok", "ciao", "openclaw", "agent:main:voice-bridge",
            ))
        finally:
            stop()

    def test_yields_content_deltas_in_order(self):
        """Happy path: three deltas + DONE → three strings out."""
        script = [
            b'data: {"choices":[{"delta":{"content":"Cia"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"o, "}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"come stai?"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        self.assertEqual(self._run(script), ["Cia", "o, ", "come stai?"])

    def test_skips_non_data_lines_and_keepalives(self):
        """Comment lines (`: ping`), blank lines, and deltas without
        content (e.g. role-only first event) must not trip the parser."""
        script = [
            b': keepalive\n\n',
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            b'\n',  # blank
            b'data: [DONE]\n\n',
        ]
        self.assertEqual(self._run(script), ["hi"])

    def test_malformed_json_lines_are_skipped(self):
        """One garbage line shouldn't poison the whole stream."""
        script = [
            b'data: not-json-at-all\n\n',
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        self.assertEqual(self._run(script), ["ok"])

    def test_empty_content_deltas_are_dropped(self):
        """`content: ""` and missing-content events must not yield empty
        strings — they'd waste a TTS roundtrip downstream."""
        script = [
            b'data: {"choices":[{"delta":{"content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        self.assertEqual(self._run(script), ["x"])

    def test_stops_at_done_sentinel(self):
        """Anything sent after `data: [DONE]` is ignored."""
        script = [
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            b'data: [DONE]\n\n',
            b'data: {"choices":[{"delta":{"content":"NOPE"}}]}\n\n',
        ]
        self.assertEqual(self._run(script), ["a"])

    def test_http_error_yields_fallback_string(self):
        """5xx → urllib raises HTTPError; the streaming wrapper catches
        and yields the same fallback `gateway_chat` returns, so the
        downstream TTS can still speak something."""
        out = self._run([], status=500)
        self.assertEqual(out, [self.vb.GATEWAY_FALLBACK_REPLY])

    def test_first_delta_arrives_before_full_response(self):
        """Streaming guarantee: a slow server that pauses 200 ms between
        chunks must yield the first delta well before the last one. If
        urllib were buffering the whole body, both deltas would land in
        the same tick and the pause would be invisible."""
        script = [
            b'data: {"choices":[{"delta":{"content":"first"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"last"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        base, stop = _start_sse_server(script, flush_pause=0.2)
        try:
            stream = self.vb.gateway_chat_stream(base, "t", "x", "openclaw")
            t0 = time.monotonic()
            first = next(stream)
            dt_first = time.monotonic() - t0
            second = next(stream)
            dt_second = time.monotonic() - t0
            with self.assertRaises(StopIteration):
                next(stream)
            self.assertEqual((first, second), ("first", "last"))
            # First delta should be effectively immediate; second must
            # come at least one flush_pause later.
            self.assertLess(dt_first, 0.15, f"first delta took {dt_first:.3f}s")
            self.assertGreater(dt_second - dt_first, 0.15,
                               f"deltas {dt_first:.3f}s vs {dt_second:.3f}s")
        finally:
            stop()


# ---------------------------------------------------------------------------
# 2. ElevenLabs.synthesize_stream — SDK monkey-patched, network-free.
# ---------------------------------------------------------------------------
class ElevenLabsSynthesizeStreamTest(unittest.TestCase):
    """We don't own the ElevenLabs websocket, so the test verifies what
    we DO own: the kwargs sent into `convert_realtime` and that audio
    chunks are passed through verbatim."""

    def _patch_sdk(self, audio_chunks):
        """Replace `elevenlabs.ElevenLabs` with a fake whose
        `text_to_speech.convert_realtime` records its kwargs and yields
        `audio_chunks`. Returns the recorder dict for assertions."""
        import elevenlabs

        record: dict = {"calls": []}

        class FakeTTS:
            def convert_realtime(self, **kw):
                # Materialize the text iterator immediately so the test
                # can see what was actually sent. In production the SDK
                # iterates lazily inside its websocket loop, but for the
                # unit test correctness is preserved either way.
                kw["text"] = list(kw["text"])
                record["calls"].append(kw)
                yield from audio_chunks

        class FakeClient:
            def __init__(self, **_kw):
                self.text_to_speech = FakeTTS()

        patcher = mock.patch.object(elevenlabs, "ElevenLabs", FakeClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        return record

    def test_passes_text_iter_and_yields_audio_chunks(self):
        from elevenlabs_voice import ElevenLabsVoice
        rec = self._patch_sdk([b"AAA", b"BBB", b"CCC"])
        el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                             tts_sample_rate=24000, tts_stream_mode="websocket")
        audio = list(el.synthesize_stream(iter(["Ciao", " mondo"])))
        self.assertEqual(audio, [b"AAA", b"BBB", b"CCC"])
        self.assertEqual(len(rec["calls"]), 1)
        call = rec["calls"][0]
        self.assertEqual(call["voice_id"], "V")
        self.assertEqual(call["model_id"], "M")
        self.assertEqual(call["output_format"], "pcm_24000")
        self.assertEqual(call["text"], ["Ciao", " mondo"])

    def test_voice_settings_promoted_to_pydantic_model(self):
        """The non-streaming path keeps voice_settings as a dict, but
        `convert_realtime` calls `.dict()` on it — so the wrapper must
        promote dict → VoiceSettings before forwarding."""
        from elevenlabs.types.voice_settings import VoiceSettings
        from elevenlabs_voice import ElevenLabsVoice
        rec = self._patch_sdk([b"x"])
        el = ElevenLabsVoice(
            api_key="x", voice_id="V", tts_model="M", tts_sample_rate=22050,
            tts_stream_mode="websocket",
            tts_voice_settings={"stability": 0.5, "similarity_boost": 0.8,
                                "use_speaker_boost": True},
        )
        list(el.synthesize_stream(iter(["hi"])))
        vs = rec["calls"][0]["voice_settings"]
        self.assertIsInstance(vs, VoiceSettings)
        self.assertEqual(vs.stability, 0.5)
        self.assertTrue(vs.use_speaker_boost)

    def test_drops_empty_chunks(self):
        """Some SDKs sometimes yield b'' as a heartbeat — we should not
        bother aplay with it."""
        self._patch_sdk([b"A", b"", b"B"])
        from elevenlabs_voice import ElevenLabsVoice
        el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                             tts_sample_rate=22050, tts_stream_mode="websocket")
        self.assertEqual(list(el.synthesize_stream(iter(["t"]))), [b"A", b"B"])

    def test_sdk_failure_swallowed_returns_empty_stream(self):
        """A websocket / auth failure must not raise out of
        `synthesize_stream` — the bridge has to keep running."""
        import elevenlabs

        class FakeTTS:
            def convert_realtime(self, **_kw):
                raise RuntimeError("websocket auth failed")
                yield  # noqa: unreachable, makes this a generator

        class FakeClient:
            def __init__(self, **_kw):
                self.text_to_speech = FakeTTS()

        with mock.patch.object(elevenlabs, "ElevenLabs", FakeClient):
            from elevenlabs_voice import ElevenLabsVoice
            el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                                 tts_sample_rate=22050, tts_stream_mode="websocket")
            # Must not raise; should yield nothing.
            self.assertEqual(list(el.synthesize_stream(iter(["x"]))), [])


class ElevenLabsSynthesizeStreamHttpSentenceTest(unittest.TestCase):
    """Default `http_sentence` mode: deltas accumulate until a sentence
    boundary, then a single `text_to_speech.stream` call is issued per
    sentence and its bytes are forwarded. We patch the SDK so no network
    is touched and we can see exactly which sentences were sent."""

    def _patch_sdk(self, chunks_by_sentence):
        """Replace `elevenlabs.ElevenLabs` with a fake whose
        `text_to_speech.stream(...)` records its kwargs and yields a
        per-call chunk list. `chunks_by_sentence` is a list of byte-lists
        — call N uses entry N (clamped to the last entry if exceeded)."""
        import elevenlabs

        record: dict = {"calls": []}

        class FakeTTS:
            def stream(self, **kw):
                idx = min(len(record["calls"]), len(chunks_by_sentence) - 1)
                record["calls"].append(kw)
                yield from chunks_by_sentence[idx]

        class FakeClient:
            def __init__(self, **_kw):
                self.text_to_speech = FakeTTS()

        patcher = mock.patch.object(elevenlabs, "ElevenLabs", FakeClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        return record

    def test_splits_at_sentence_boundaries_and_yields_per_sentence(self):
        """`Ciao, come stai? Sto bene.` arrives across multiple deltas →
        two `stream()` calls, audio in delta order, chunks forwarded."""
        rec = self._patch_sdk([[b"A1", b"A2"], [b"B1"]])
        from elevenlabs_voice import ElevenLabsVoice
        el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                             tts_sample_rate=24000)
        deltas = ["Ciao, come ", "stai? ", "Sto", " bene."]
        out = list(el.synthesize_stream(iter(deltas)))
        self.assertEqual(out, [b"A1", b"A2", b"B1"])
        self.assertEqual(
            [c["text"] for c in rec["calls"]],
            ["Ciao, come stai?", "Sto bene."],
        )
        # Sample rate / format / model contract still pinned.
        self.assertEqual(rec["calls"][0]["voice_id"], "V")
        self.assertEqual(rec["calls"][0]["model_id"], "M")
        self.assertEqual(rec["calls"][0]["output_format"], "pcm_24000")

    def test_residual_text_without_terminator_is_flushed_at_end(self):
        """A reply that ends mid-sentence (e.g. token cap) should still
        be spoken — the buffer's tail flushes after the iterator ends."""
        rec = self._patch_sdk([[b"PCM"]])
        from elevenlabs_voice import ElevenLabsVoice
        el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                             tts_sample_rate=22050)
        out = list(el.synthesize_stream(iter(["Senza punteggiatura finale"])))
        self.assertEqual(out, [b"PCM"])
        self.assertEqual([c["text"] for c in rec["calls"]],
                         ["Senza punteggiatura finale"])

    def test_forwards_language_and_text_normalization(self):
        """Unlike the WS path, the HTTP endpoint accepts `language_code`
        and `apply_text_normalization`; the wrapper must forward them."""
        rec = self._patch_sdk([[b"x"]])
        from elevenlabs_voice import ElevenLabsVoice
        el = ElevenLabsVoice(
            api_key="x", voice_id="V", tts_model="M", tts_sample_rate=22050,
            tts_language="it", tts_text_normalization="on",
            tts_voice_settings={"stability": 0.4},
        )
        list(el.synthesize_stream(iter(["Ciao."])))
        call = rec["calls"][0]
        self.assertEqual(call["language_code"], "it")
        self.assertEqual(call["apply_text_normalization"], "on")
        # voice_settings forwarded as-is (dict), matching the
        # non-streaming `synthesize()` behavior.
        self.assertEqual(call["voice_settings"], {"stability": 0.4})

    def test_per_sentence_failure_does_not_break_remaining_sentences(self):
        """If `stream()` blows up on one sentence, log + skip; the next
        sentence still gets its chance."""
        import elevenlabs

        class FakeTTS:
            def __init__(self):
                self.calls = 0

            def stream(self, **kw):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("transient")
                    yield  # noqa: unreachable, makes this a generator
                yield b"OK"

        class FakeClient:
            def __init__(self, **_kw):
                self.text_to_speech = FakeTTS()

        with mock.patch.object(elevenlabs, "ElevenLabs", FakeClient):
            from elevenlabs_voice import ElevenLabsVoice
            el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                                 tts_sample_rate=22050)
            out = list(el.synthesize_stream(iter(["Prima. Seconda."])))
            self.assertEqual(out, [b"OK"])

    def test_empty_input_makes_no_sdk_calls(self):
        rec = self._patch_sdk([[b"x"]])
        from elevenlabs_voice import ElevenLabsVoice
        el = ElevenLabsVoice(api_key="x", voice_id="V", tts_model="M",
                             tts_sample_rate=22050)
        self.assertEqual(list(el.synthesize_stream(iter([]))), [])
        self.assertEqual(rec["calls"], [])

    def test_unknown_stream_mode_rejected_at_construction(self):
        from elevenlabs_voice import ElevenLabsVoice
        with self.assertRaises(ValueError):
            ElevenLabsVoice(api_key="x", voice_id="V", tts_stream_mode="grpc")


# ---------------------------------------------------------------------------
# 3. Deepgram.synthesize_stream — fallback shim.
# ---------------------------------------------------------------------------
class DeepgramSynthesizeStreamTest(unittest.TestCase):
    """Deepgram has no incremental TTS path on the REST endpoint we use;
    the wrapper buffers the text and yields one PCM blob — the test
    pins that contract so a future "streaming" rewrite is opt-in, not
    accidental."""

    def test_buffers_text_and_yields_single_chunk(self):
        from deepgram_voice import DeepgramVoice
        dg = DeepgramVoice(api_key="x")
        with mock.patch.object(dg, "synthesize", return_value=b"PCMDATA") as syn:
            out = list(dg.synthesize_stream(iter(["Hello, ", "world!"])))
            self.assertEqual(out, [b"PCMDATA"])
            syn.assert_called_once_with("Hello, world!")

    def test_empty_text_yields_nothing_and_skips_call(self):
        from deepgram_voice import DeepgramVoice
        dg = DeepgramVoice(api_key="x")
        with mock.patch.object(dg, "synthesize", return_value=b"X") as syn:
            self.assertEqual(list(dg.synthesize_stream(iter(["", "  ", "\n"]))), [])
            syn.assert_not_called()

    def test_synthesize_returning_none_yields_nothing(self):
        """When the underlying TTS errors out (synthesize returns None),
        the stream must end cleanly rather than yielding `None` as bytes."""
        from deepgram_voice import DeepgramVoice
        dg = DeepgramVoice(api_key="x")
        with mock.patch.object(dg, "synthesize", return_value=None):
            self.assertEqual(list(dg.synthesize_stream(iter(["hi"]))), [])


# ---------------------------------------------------------------------------
# 4. play_audio_stream — Popen replaced with a tape recorder.
# ---------------------------------------------------------------------------
class _FakePopen:
    """Stand-in for `subprocess.Popen` returned objects.

    Records the `argv` passed in, captures everything written to stdin
    (in order, with timestamps so we can prove streaming behavior), and
    behaves enough like a real Popen for `play_audio_stream` to drive it.
    """

    instances: list["_FakePopen"] = []

    def __init__(self, argv, stdin=None, stderr=None, bufsize=None):
        self.argv = argv
        self.bufsize = bufsize
        self.writes: list[tuple[float, bytes]] = []
        self.closed = False
        self.waited = False
        self._t0 = time.monotonic()
        self.broken_after: int | None = None  # raise BrokenPipeError after N writes
        self.stdin = self._make_stdin()
        _FakePopen.instances.append(self)

    def _make_stdin(self):
        outer = self

        class _Stdin:
            def write(self, data):
                if outer.broken_after is not None and len(outer.writes) >= outer.broken_after:
                    raise BrokenPipeError("aplay went away")
                outer.writes.append((time.monotonic() - outer._t0, data))

            def close(self):
                outer.closed = True

        return _Stdin()

    def wait(self):
        self.waited = True
        return 0


class PlayAudioStreamTest(unittest.TestCase):
    def setUp(self):
        self.vb = _load_voice_bridge()
        _FakePopen.instances.clear()
        self._popen_patch = mock.patch.object(subprocess, "Popen", _FakePopen)
        self._popen_patch.start()
        self.addCleanup(self._popen_patch.stop)

    def _last_proc(self) -> _FakePopen:
        self.assertEqual(len(_FakePopen.instances), 1, "expected exactly one aplay")
        return _FakePopen.instances[0]

    def test_invokes_aplay_with_expected_args(self):
        """The subprocess argv must keep S16_LE / mono / configured rate.
        These three are the contract with both the TTS output_format and
        the on-host ALSA dmix; drift here = silent garbage at playback."""
        self.vb.play_audio_stream("plug:jabra_dmix", iter([b"x"]), 22050)
        proc = self._last_proc()
        self.assertEqual(proc.argv[0], "aplay")
        self.assertIn("-f", proc.argv)
        self.assertIn("S16_LE", proc.argv)
        self.assertIn("-r", proc.argv)
        self.assertIn("22050", proc.argv)
        self.assertIn("-c", proc.argv)
        self.assertIn("1", proc.argv)
        self.assertIn("-D", proc.argv)
        self.assertIn("plug:jabra_dmix", proc.argv)

    def test_writes_each_chunk_in_order(self):
        chunks = [b"AAA", b"BBBB", b"CC"]
        result = self.vb.play_audio_stream("dev", iter(chunks), 24000)
        self.assertTrue(result)
        proc = self._last_proc()
        self.assertEqual([data for _, data in proc.writes], chunks)
        self.assertTrue(proc.closed)
        self.assertTrue(proc.waited)

    def test_zero_chunks_returns_false_but_still_drains(self):
        """An empty stream is normal (e.g. error path that didn't yield
        a fallback). aplay still gets started and stopped cleanly."""
        result = self.vb.play_audio_stream("dev", iter([]), 24000)
        self.assertFalse(result)
        proc = self._last_proc()
        self.assertEqual(proc.writes, [])
        self.assertTrue(proc.closed)
        self.assertTrue(proc.waited)

    def test_first_chunk_written_before_later_chunks_produced(self):
        """The whole point of streaming. We give play_audio_stream an
        iterator that pauses 150 ms before the second chunk; the first
        chunk's stdin write must land BEFORE that pause, not after."""
        def slow():
            yield b"FIRST"
            time.sleep(0.15)
            yield b"SECOND"

        self.vb.play_audio_stream("dev", slow(), 22050)
        proc = self._last_proc()
        self.assertEqual([d for _, d in proc.writes], [b"FIRST", b"SECOND"])
        t_first, t_second = proc.writes[0][0], proc.writes[1][0]
        # First write essentially at t≈0; second at >= 150 ms.
        self.assertLess(t_first, 0.05, f"first write delayed: {t_first:.3f}s")
        self.assertGreater(t_second - t_first, 0.1,
                           f"writes too close: {t_first:.3f}s vs {t_second:.3f}s")

    def test_broken_pipe_stops_pulling_but_returns_cleanly(self):
        """If aplay dies mid-playback we must NOT explode the bridge —
        BrokenPipeError on write should break out of the loop quietly,
        and the function should still close the pipe and reap the proc."""
        pulled: list[bytes] = []

        def src():
            for c in (b"A", b"B", b"C", b"D"):
                pulled.append(c)
                yield c

        # First write succeeds; second raises BrokenPipeError.
        with mock.patch.object(_FakePopen, "broken_after", 1, create=True):
            # Configure on the *next* instance via class default.
            pass

        # Configure the broken_after on the instance after construction
        # by hooking the constructor.
        orig_init = _FakePopen.__init__

        def patched_init(self, *a, **kw):
            orig_init(self, *a, **kw)
            self.broken_after = 1

        with mock.patch.object(_FakePopen, "__init__", patched_init):
            result = self.vb.play_audio_stream("dev", src(), 22050)

        proc = self._last_proc()
        # Exactly one chunk made it in before the pipe broke.
        self.assertEqual([d for _, d in proc.writes], [b"A"])
        # The "wrote_any" return value reflects the successful write.
        self.assertTrue(result)
        # The function must have stopped pulling after the failure
        # (so the upstream TTS websocket isn't held open forever).
        self.assertEqual(pulled, [b"A", b"B"])
        self.assertTrue(proc.closed)
        self.assertTrue(proc.waited)


# ---------------------------------------------------------------------------
# 5. End-to-end: real loopback gateway → fake TTS → fake aplay.
# ---------------------------------------------------------------------------
class _FakeTTSProvider:
    """Drop-in for a voice provider's `.synthesize_stream`.

    Yields one PCM blob per text delta so the test can correlate
    ordering between the gateway stream and the audio stream — i.e. the
    audio for delta N must reach aplay before the audio for delta N+1.
    """

    def __init__(self):
        self.text_seen: list[str] = []

    def synthesize_stream(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        for delta in text_iter:
            self.text_seen.append(delta)
            yield b"<" + delta.encode() + b">"


class EndToEndStreamingTest(unittest.TestCase):
    """Glue test: with all three real wrappers (gateway_chat_stream +
    a fake provider satisfying the contract + play_audio_stream), do the
    bytes that hit aplay reflect the gateway deltas in the right order?"""

    def setUp(self):
        self.vb = _load_voice_bridge()
        _FakePopen.instances.clear()
        self._popen_patch = mock.patch.object(subprocess, "Popen", _FakePopen)
        self._popen_patch.start()
        self.addCleanup(self._popen_patch.stop)

    def test_full_pipeline_delivers_chunks_in_delta_order(self):
        script = [
            b'data: {"choices":[{"delta":{"content":"uno"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" due"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" tre"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        base, stop = _start_sse_server(script, flush_pause=0.05)
        try:
            tts = _FakeTTSProvider()
            text_stream = self.vb.gateway_chat_stream(base, "t", "x", "openclaw")
            audio_stream = tts.synthesize_stream(text_stream)
            played = self.vb.play_audio_stream("dev", audio_stream, 22050)
        finally:
            stop()

        self.assertTrue(played)
        self.assertEqual(tts.text_seen, ["uno", " due", " tre"])
        proc = _FakePopen.instances[0]
        self.assertEqual(
            [d for _, d in proc.writes],
            [b"<uno>", b"< due>", b"< tre>"],
        )

    def test_first_audio_reaches_aplay_before_last_gateway_delta(self):
        """The streaming guarantee end to end: aplay must see the first
        audio chunk well before the gateway has finished sending. With
        a 200 ms pause between deltas, the first stdin write should be
        very close to the first delta's arrival, not after the third."""
        script = [
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        base, stop = _start_sse_server(script, flush_pause=0.2)
        try:
            tts = _FakeTTSProvider()
            text_stream = self.vb.gateway_chat_stream(base, "t", "x", "openclaw")
            audio_stream = tts.synthesize_stream(text_stream)
            t0 = time.monotonic()
            self.vb.play_audio_stream("dev", audio_stream, 22050)
            t_end = time.monotonic() - t0
        finally:
            stop()

        proc = _FakePopen.instances[0]
        self.assertEqual(len(proc.writes), 3)
        t_first = proc.writes[0][0]
        t_last = proc.writes[-1][0]
        # First audio reached aplay quickly...
        self.assertLess(t_first, 0.15, f"first audio took {t_first:.3f}s")
        # ...well before the whole reply finished arriving.
        self.assertGreater(t_last - t_first, 0.3,
                           f"only {t_last - t_first:.3f}s between first and last")
        # And the entire run was bounded by the flush pauses, not blown
        # up by some accidental buffering layer.
        self.assertLess(t_end, 1.5)

    def test_gateway_error_still_produces_audible_fallback(self):
        """If the gateway returns 5xx, gateway_chat_stream yields the
        fallback string; the TTS stream then turns it into audio; the
        user hears the apology rather than dead silence."""
        base, stop = _start_sse_server([], status=500)
        try:
            tts = _FakeTTSProvider()
            text_stream = self.vb.gateway_chat_stream(base, "t", "x", "openclaw")
            audio_stream = tts.synthesize_stream(text_stream)
            self.vb.play_audio_stream("dev", audio_stream, 22050)
        finally:
            stop()

        self.assertEqual(tts.text_seen, [self.vb.GATEWAY_FALLBACK_REPLY])
        proc = _FakePopen.instances[0]
        self.assertEqual(len(proc.writes), 1)
        self.assertIn(b"problema di connessione", proc.writes[0][1])


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
