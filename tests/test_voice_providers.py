#!/usr/bin/env python3
"""Smoke tests for the voice provider wrappers.

These don't hit the real APIs — they verify the public interface matches
what voice-bridge.py expects, and that the safe-fallback paths (empty
input, no text) behave correctly without making outbound calls.

Run: .venv/bin/python test_voice_providers.py
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepgram_voice import DeepgramVoice  # noqa: E402
from elevenlabs_voice import ElevenLabsVoice  # noqa: E402


class InterfaceContractTest(unittest.TestCase):
    """Both providers must expose the same `transcribe` / `synthesize`
    surface, so swapping one for the other in voice-bridge.py is a
    one-line constructor change."""

    def test_both_providers_expose_transcribe_and_synthesize(self) -> None:
        for cls in (DeepgramVoice, ElevenLabsVoice):
            self.assertTrue(hasattr(cls, "transcribe"), f"{cls.__name__} missing transcribe")
            self.assertTrue(hasattr(cls, "synthesize"), f"{cls.__name__} missing synthesize")
            self.assertTrue(callable(getattr(cls, "transcribe")))
            self.assertTrue(callable(getattr(cls, "synthesize")))


class DeepgramVoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dg = DeepgramVoice(api_key="dummy", stt_language="it")

    def test_empty_pcm_returns_empty_string_without_network(self) -> None:
        # Should short-circuit before making any HTTP call.
        self.assertEqual(self.dg.transcribe(b"", 16000), "")

    def test_empty_text_returns_none_without_network(self) -> None:
        self.assertIsNone(self.dg.synthesize(""))

    def test_dummy_key_transcribe_returns_empty_on_auth_error(self) -> None:
        # Fake key + tiny PCM — must not raise; the broad except in
        # transcribe() converts API errors to "" so the bridge can
        # log "No speech detected" and move on.
        result = self.dg.transcribe(b"\x00\x00" * 1600, 16000)
        self.assertEqual(result, "")


class ElevenLabsVoiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.el = ElevenLabsVoice(api_key="dummy", voice_id="dummy_voice")

    def test_empty_pcm_returns_empty_string_without_network(self) -> None:
        self.assertEqual(self.el.transcribe(b"", 16000), "")

    def test_empty_text_returns_none_without_network(self) -> None:
        self.assertIsNone(self.el.synthesize(""))

    def test_dummy_key_synthesize_returns_none_on_auth_error(self) -> None:
        result = self.el.synthesize("ciao")
        self.assertIsNone(result)


class DefaultsTest(unittest.TestCase):
    """Pin the operationally-important defaults so an accidental edit
    doesn't silently change Deepgram from Italian or ElevenLabs from
    raw PCM."""

    def test_deepgram_defaults_to_italian_nova3(self) -> None:
        dg = DeepgramVoice(api_key="x")
        self.assertEqual(dg._stt_model, "nova-3")
        self.assertEqual(dg._stt_language, "it")

    def test_elevenlabs_output_format_derived_from_sample_rate(self) -> None:
        # The wrapper asks ElevenLabs for raw PCM at the configured rate.
        # If the format string ever drifts to MP3/WAV again, aplay (which
        # consumes raw PCM) plays silent garbage.
        self.assertEqual(ElevenLabsVoice(api_key="x", voice_id="v")._tts_output_format, "pcm_22050")
        self.assertEqual(
            ElevenLabsVoice(api_key="x", voice_id="v", tts_sample_rate=44100)._tts_output_format,
            "pcm_44100",
        )

    def test_elevenlabs_stt_language_default(self) -> None:
        el = ElevenLabsVoice(api_key="x", voice_id="v")
        self.assertEqual(el._stt_language, "ita")


class ProviderSelectionTest(unittest.TestCase):
    """voice-bridge.json must be honored: changing the stt_provider /
    tts_provider keys swaps which class voice-bridge.py instantiates
    for each role."""

    def _load_config_with_json(self, **overrides) -> dict:
        """Run voice-bridge.py's load_config() against a temp JSON."""
        import importlib.util
        import json as _json
        import tempfile

        base = {
            "gateway_base_url": "http://127.0.0.1:18789",
            "sample_rate": 16000,
            "tts_sample_rate": 44100,
            "chunk_size": 1024,
            "silence_timeout_ms": 1500,
            "max_recording_s": 30,
            "output_device": "plug:jabra_dmix",
            "hid_mute_enabled": True,
        }
        base.update(overrides)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            _json.dump(base, f)
            tmp = f.name
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "vb", os.path.join(here, "voice-bridge.py")
            )
            vb = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vb)
            vb.CONFIG_PATH = tmp
            return vb.load_config()
        finally:
            os.unlink(tmp)

    def test_default_when_keys_missing(self) -> None:
        cfg = self._load_config_with_json()
        self.assertEqual(cfg["stt_provider"], "elevenlabs")
        self.assertEqual(cfg["tts_provider"], "elevenlabs")

    def test_mixed_providers_parsed(self) -> None:
        cfg = self._load_config_with_json(stt_provider="deepgram", tts_provider="elevenlabs")
        self.assertEqual(cfg["stt_provider"], "deepgram")
        self.assertEqual(cfg["tts_provider"], "elevenlabs")

    def test_tts_sample_rate_propagated(self) -> None:
        cfg = self._load_config_with_json(tts_sample_rate=48000)
        self.assertEqual(cfg["tts_sample_rate"], 48000)

    def test_unknown_provider_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load_config_with_json(stt_provider="whisper")


class DeepgramOptionsTest(unittest.TestCase):
    """The kwargs built for Deepgram's transcribe_file must respect
    openclaw's providerOptions — particularly `detect_language=True`
    must override the class's default `stt_language="it"` so the API
    auto-detects instead of being pinned to Italian."""

    def _kwargs_built_by(self, **init_kwargs) -> dict:
        """Build the same kwargs dict transcribe_file would get."""
        from deepgram_voice import DeepgramVoice
        dg = DeepgramVoice(api_key="x", **init_kwargs)
        # Reproduce the merge logic from _transcribe_async without
        # making a network call.
        kwargs: dict = {
            "request": b"<wav>",
            "model": dg._stt_model,
            "smart_format": True,
        }
        kwargs.update(dg._stt_options)
        if (
            dg._stt_language is not None
            and "language" not in kwargs
            and not kwargs.get("detect_language")
        ):
            kwargs["language"] = dg._stt_language
        return kwargs

    def test_default_pins_italian_language(self) -> None:
        kw = self._kwargs_built_by()
        self.assertEqual(kw.get("language"), "it")
        self.assertNotIn("detect_language", kw)

    def test_detect_language_drops_explicit_language(self) -> None:
        # Mirrors what openclaw config yields.
        kw = self._kwargs_built_by(stt_options={"detect_language": True, "punctuate": True})
        self.assertNotIn("language", kw, "language must be dropped when detect_language=True")
        self.assertTrue(kw.get("detect_language"))
        self.assertTrue(kw.get("punctuate"))

    def test_explicit_language_in_options_wins_over_class_default(self) -> None:
        kw = self._kwargs_built_by(stt_language="it", stt_options={"language": "fr"})
        self.assertEqual(kw["language"], "fr")

    def test_options_can_override_smart_format(self) -> None:
        kw = self._kwargs_built_by(stt_options={"smart_format": False})
        self.assertFalse(kw["smart_format"])


class ProviderFactoryTest(unittest.TestCase):
    """_build_voice_provider must return the right class for each name."""

    def _vb(self):
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location(
            "vb", os.path.join(here, "voice-bridge.py")
        )
        vb = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vb)
        return vb

    def test_factory_returns_elevenlabs_when_selected(self) -> None:
        vb = self._vb()
        cfg = {
            "stt_provider": "elevenlabs",
            "elevenlabs_key": "x", "elevenlabs_voice": "v",
            "elevenlabs_model": "eleven_multilingual_v2",
            "deepgram_key": "x",
            "tts_sample_rate": 44100,
        }
        provider = vb._build_voice_provider("stt", cfg)
        self.assertIsInstance(provider, ElevenLabsVoice)

    def test_factory_returns_deepgram_when_selected(self) -> None:
        vb = self._vb()
        cfg = {
            "tts_provider": "deepgram",
            "elevenlabs_key": "x", "elevenlabs_voice": "v",
            "elevenlabs_model": "eleven_multilingual_v2",
            "deepgram_key": "x",
            "tts_sample_rate": 44100,
        }
        provider = vb._build_voice_provider("tts", cfg)
        self.assertIsInstance(provider, DeepgramVoice)


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=2).result.wasSuccessful() else 1)
