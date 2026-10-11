#!/usr/bin/env python3
"""Speaker identification and the "far conoscenza" wizard.

No model and no audio: `SpeakerId.embed` is replaced by fixed vectors. Covers
the profile store, the wizard's state machine (`Enrollment`) and how the
bridge drives it — started by its phrase, takes kept away from the agent,
cancelled by voice, silence or the HID button — plus the speaker tag on
every message to the agent.

Run: .venv/bin/python tests/test_speaker_id.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from unittest import mock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import speaker_id  # noqa: E402


def _load_voice_bridge():
    spec = importlib.util.spec_from_file_location(
        "voice_bridge", os.path.join(_PROJECT_ROOT, "voice-bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VB = _load_voice_bridge()
S = VB.State

ANNA = [1.0, 0.0, 0.0]
ANNA_ISH = speaker_id._unit([0.9, 0.2, 0.0])
BRUNO = [0.0, 1.0, 0.0]
PCM_3S = bytes(16000 * 2 * 3)
PCM_1S = bytes(16000 * 2)
# Anna in the room: 0.60 and 0.45 against her enrolled voiceprint.
ANNA_FAR = speaker_id._unit([0.6, 0.8, 0.0])
ANNA_FARTHER = speaker_id._unit([0.45, 0.89, 0.0])


def _scfg(tmp: str, **over) -> dict:
    return speaker_id.config({"speaker_id": {
        "enabled": True, "model_dir": tmp,
        "profiles_path": os.path.join(tmp, "speakers.json"),
        "sentences": ["Uno.", "Due.", "Tre."], "far_sentences": [], **over}})


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.tmp = self._dir.name
        self.scfg = _scfg(self.tmp)
        self.sid = speaker_id.SpeakerId(self.scfg)
        self.sid.warm = lambda: None


class NameTest(unittest.TestCase):
    def test_parse(self):
        for said, name in [("Mi chiamo Anna.", "Anna"), ("Anna", "Anna"),
                           ("Ciao, io sono Gian Luca!", "Gian Luca"),
                           ("Sono Bruno", "Bruno"), ("", ""),
                           ("Non lo so davvero cosa dire", "")]:
            self.assertEqual(speaker_id._parse_name(said), name, said)


class ProfilesTest(_Tmp):
    def test_identify_known_unknown_and_reload(self):
        self.assertEqual(self.sid.identify(PCM_3S, 16000), "Sconosciuto")  # nobody enrolled
        self.sid.save_profile("Anna", [ANNA, ANNA_ISH])
        self.sid.save_profile("Bruno", [BRUNO])
        again = speaker_id.SpeakerId(self.scfg)
        self.assertEqual(set(again.profiles), {"Anna", "Bruno"})
        for voice, name in [(ANNA, "Anna"), (BRUNO, "Bruno"), ([0.0, 0.0, 1.0], "Sconosciuto")]:
            again.embed = lambda *_a, voice=voice: voice
            self.assertEqual(again.identify(PCM_3S, 16000), name)
        self.assertEqual(again.tag("Anna"), "[Sta parlando: Anna] ")
        self.assertEqual(again.tag(None), "[Sta parlando: Sconosciuto] ")

    def _hears(self, sid, voice, pcm=PCM_3S) -> str:
        sid.embed = lambda *_a: voice
        return sid.identify(pcm, 16000)

    def test_short_follow_up_keeps_the_conversation_speaker(self):
        self.sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(self.sid, BRUNO, PCM_1S), "Sconosciuto")  # nobody yet
        self.assertEqual(self._hears(self.sid, ANNA), "Anna")
        self.assertEqual(self._hears(self.sid, BRUNO, PCM_1S), "Anna")  # "ok"
        self.sid.end_conversation()  # idle / mute
        self.assertEqual(self._hears(self.sid, BRUNO, PCM_1S), "Sconosciuto")

    def test_long_unknown_utterance_ends_the_memory(self):
        self.sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(self.sid, ANNA), "Anna")
        self.assertEqual(self._hears(self.sid, BRUNO), "Sconosciuto")  # 3 s: somebody else
        self.assertEqual(self._hears(self.sid, BRUNO, PCM_1S), "Sconosciuto")

    def test_another_known_voice_takes_over(self):
        self.sid.save_profile("Anna", [ANNA])
        self.sid.save_profile("Bruno", [BRUNO])
        self.assertEqual(self._hears(self.sid, ANNA), "Anna")
        self.assertEqual(self._hears(self.sid, BRUNO, PCM_1S), "Bruno")
        self.assertEqual(self._hears(self.sid, [0.0, 0.0, 1.0], PCM_1S), "Bruno")

    def test_memory_off(self):
        sid = speaker_id.SpeakerId(_scfg(self.tmp, carry_max_s=0))
        sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(self._hears(sid, BRUNO, PCM_1S), "Sconosciuto")

    def _adaptive(self, **over):
        return speaker_id.SpeakerId(_scfg(self.tmp, adapt={
            "enabled": True, "min_samples": 2, **over}))

    def test_adapted_voiceprint_learns_from_a_confident_conversation(self):
        sid = self._adaptive()
        sid.save_profile("Anna", [ANNA])
        # Without a confident turn nothing is learned.
        self.assertEqual(self._hears(sid, ANNA_FAR), "Anna")
        sid.end_conversation()
        self.assertEqual(sid._samples.get("Anna", []), [])
        # A turn heard before the confident one counts once it comes.
        self.assertEqual(self._hears(sid, ANNA_FAR), "Anna")
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(len(sid._samples["Anna"]), 2)
        sid.end_conversation()
        self.assertEqual(self._hears(sid, ANNA_FARTHER), "Anna")  # 0.45 enrolled, 0.80 adapted
        # Saved, and only used when adapt is on.
        self.assertEqual(self._hears(self._adaptive(), ANNA_FARTHER), "Anna")
        self.assertEqual(self._hears(speaker_id.SpeakerId(self.scfg), ANNA_FARTHER),
                         "Sconosciuto")
        # Enrolling again keeps the samples that are still like the voice...
        sid.save_profile("Anna", [ANNA_ISH])
        self.assertEqual(len(self._adaptive()._samples["Anna"]), 2)
        self.assertEqual(self._hears(self._adaptive(), ANNA_FARTHER), "Anna")
        # ...and drops the others.
        sid.save_profile("Anna", [[0.0, 0.0, 1.0]])
        self.assertEqual(self._adaptive()._samples["Anna"], [])

    def test_short_unknown_turn_counts_once_a_confident_one_follows(self):
        sid = self._adaptive(min_samples=5)
        sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(sid, ANNA_FARTHER, PCM_1S), "Sconosciuto")  # from afar
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(len(sid._samples["Anna"]), 2)
        sid.end_conversation()
        # A long one that matched nobody is somebody else: never a sample.
        self.assertEqual(self._hears(sid, ANNA_FARTHER), "Sconosciuto")
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(len(sid._samples["Anna"]), 3)

    def test_adapted_score_is_the_closest_samples_not_their_mean(self):
        sid = self._adaptive(min_samples=5, top_k=2)
        sid.save_profile("Anna", [ANNA])
        self._hears(sid, ANNA)
        for voice in [ANNA, ANNA, ANNA, ANNA_FAR, ANNA_FAR]:
            self._hears(sid, voice)
        sid.end_conversation()
        sid.embed = lambda *_a: ANNA_FARTHER
        m = sid.match(PCM_3S, 16000)
        self.assertAlmostEqual(m.enrolled, 0.45, places=2)
        self.assertGreater(m.adapted, 0.95)  # the two far samples; the mean gives 0.72
        self.assertLess(speaker_id._cosine(ANNA_FARTHER, speaker_id._mean(sid._samples["Anna"])), 0.75)

    def test_far_voiceprint_is_a_second_enrolled_one(self):
        self.sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(self.sid, ANNA_FARTHER), "Sconosciuto")
        self.sid.save_far("Anna", [ANNA_FAR, ANNA_FARTHER])
        again = speaker_id.SpeakerId(self.scfg)
        self.assertEqual(self._hears(again, ANNA_FARTHER), "Anna")
        self.assertEqual(self._hears(again, ANNA), "Anna")
        self.assertEqual(again.profiles["Anna"], ANNA)  # not averaged in
        with self.assertRaises(OSError):
            self.sid.save_far("Bruno", [BRUNO])

    def test_samples_must_be_long_and_close_enough(self):
        sid = self._adaptive()
        sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(self._hears(sid, ANNA_FAR, bytes(16000)), "Anna")  # 0.5 s
        self.assertEqual(self._hears(sid, BRUNO, PCM_1S), "Anna")  # carried, 0.00
        self.assertEqual(len(sid._samples["Anna"]), 1)

    def test_adapted_voiceprint_that_drifts_is_dropped(self):
        sid = self._adaptive(min_anchor_similarity=0.95, min_score=0.0)
        sid.save_profile("Anna", [ANNA])
        self.assertEqual(self._hears(sid, ANNA), "Anna")
        self.assertEqual(self._hears(sid, BRUNO, PCM_1S), "Anna")  # carried: a wrong sample
        self.assertEqual(sid._samples["Anna"], [])
        self.assertNotIn("Anna", sid._adapted)
        self.assertEqual(self._adaptive()._samples["Anna"], [])

    def test_profiles_of_another_model_are_ignored(self):
        self.sid.save_profile("Anna", [ANNA])
        other = speaker_id.SpeakerId(_scfg(self.tmp, model_url="http://x/other.onnx"))
        self.assertEqual(other.profiles, {})


class EnrollmentTest(_Tmp):
    def setUp(self):
        super().setUp()
        self.en = speaker_id.Enrollment(self.sid, self.scfg)

    def test_request_phrase(self):
        self.assertTrue(self.en.is_request("Facciamo conoscenza!"))
        self.assertFalse(self.en.is_request("Metti la musica"))

    def test_full_session(self):
        step = self.en.start()
        self.assertEqual(step, speaker_id.Step([speaker_id.INTRO, speaker_id.ASK_NAME], True))
        self.assertFalse(self.en.wants_voice)
        step = self.en.on_utterance("Mi chiamo Anna.", None, 1.0)
        self.assertEqual(step.say[0], ("tts", "[cheerful] Piacere, Anna!", speaker_id.GREET))
        self.assertEqual(step.say[1], speaker_id.take_prompt("Uno."))
        self.assertTrue(self.en.wants_voice)
        step = self.en.on_utterance("Uno.", ANNA, 3.0)
        self.assertEqual(step.say[1], speaker_id.take_prompt("Due."))
        step = self.en.on_utterance("Due.", ANNA_ISH, 3.0)
        self.assertEqual(step.say[1], speaker_id.take_prompt("Tre."))
        step = self.en.on_utterance("Tre.", ANNA, 3.0)
        self.assertEqual(step, speaker_id.Step([speaker_id.DONE], False))
        self.assertFalse(self.en.active)
        self.assertIn("Anna", speaker_id.SpeakerId(self.scfg).profiles)

    def _far_wizard(self):
        scfg = _scfg(self.tmp, far_sentences=["Quattro.", "Cinque."])
        sid = speaker_id.SpeakerId(scfg)
        sid.warm = lambda: None
        return sid, speaker_id.Enrollment(sid, scfg)

    def test_far_takes_follow_the_close_ones(self):
        sid, en = self._far_wizard()
        en.start("Anna")
        en.on_utterance("Uno.", ANNA, 3.0)
        en.on_utterance("Due.", ANNA, 3.0)
        step = en.on_utterance("Tre.", ANNA, 3.0)
        self.assertEqual(step.say[1:], [speaker_id.FAR_INTRO, speaker_id.take_prompt("Quattro.")])
        self.assertTrue(step.listen and en.wants_voice)
        self.assertIn("Anna", sid.profiles)  # the close voiceprint is already saved
        step = en.on_utterance("Quattro.", BRUNO, 3.0)  # not the same voice
        self.assertEqual(step.say, [speaker_id.UNCLEAR, speaker_id.take_prompt("Quattro.")])
        step = en.on_utterance("Quattro.", ANNA_FAR, 3.0)
        self.assertEqual(step.say[1], speaker_id.take_prompt("Cinque."))
        step = en.on_utterance("Cinque.", ANNA_FARTHER, 3.0)
        self.assertEqual(step, speaker_id.Step([speaker_id.FAR_DONE], False))
        self.assertIn("Anna", sid._far)

    def test_a_take_must_be_the_sentence_asked(self):
        sid, en = self._far_wizard()
        en.start("Anna")
        step = en.on_utterance("Che ore sono?", ANNA, 3.0)  # something else
        self.assertEqual(step.say, [speaker_id.UNCLEAR, speaker_id.take_prompt("Uno.")])
        for said in ("uno", "Due!", "Tre."):
            en.on_utterance(said, ANNA, 3.0)
        # From too far the STT hears little or nothing: come closer.
        for said in ("", "[rumore di sottofondo]"):
            step = en.on_utterance(said, ANNA_FAR, 3.0)
            self.assertEqual(step.say, [speaker_id.FAR_CLOSER, speaker_id.take_prompt("Quattro.")])
        self.assertEqual(en.on_utterance("Quattro", ANNA_FAR, 3.0).say[1],
                         speaker_id.take_prompt("Cinque."))
        # Not checked when turned off.
        en2 = speaker_id.Enrollment(sid, _scfg(self.tmp, take_text_match=0))
        en2.start("Bruno")
        self.assertEqual(en2.on_utterance("Che ore sono?", BRUNO, 3.0).say[1],
                         speaker_id.take_prompt("Due."))

    def test_far_takes_can_be_skipped(self):
        for stop in (lambda en: en.on_utterance("Basta.", None, 0.5),
                     lambda en: en.cancel("timeout")):
            sid, en = self._far_wizard()
            en.start("Anna")
            for said in ("Uno.", "Due.", "Tre."):
                en.on_utterance(said, ANNA, 3.0)
            self.assertEqual(stop(en), speaker_id.Step([speaker_id.FAR_SKIPPED], False))
            self.assertIn("Anna", sid.profiles)
            self.assertNotIn("Anna", sid._far)

    def test_far_takes_alone_for_a_known_voice(self):
        sid, en = self._far_wizard()
        self.assertTrue(en.is_far_request("Impara la mia voce da lontano."))
        self.assertFalse(en.is_far_request("Facciamo conoscenza!"))
        sid.save_profile("Anna", [ANNA])
        step = en.start("Anna", far_only=True)
        self.assertEqual(step.say, [speaker_id.FAR_ONLY_INTRO, speaker_id.take_prompt("Quattro.")])
        en.on_utterance("Quattro.", ANNA_FAR, 3.0)
        self.assertEqual(en.on_utterance("Cinque.", ANNA_FAR, 3.0).say, [speaker_id.FAR_DONE])
        self.assertEqual(sid.profiles["Anna"], ANNA)  # untouched
        # Asked by a voice nobody knows: the name, then the whole wizard.
        step = en.start(far_only=True)
        self.assertEqual(step.say, [speaker_id.INTRO, speaker_id.ASK_NAME])
        step = en.on_utterance("Mi chiamo Bruno.", None, 1.5)
        self.assertEqual(step.say[1], speaker_id.take_prompt("Uno."))
        # ...and the name of a known one: its far takes.
        en.cancel("test")
        en.start(far_only=True)
        step = en.on_utterance("Anna.", None, 1.0)
        self.assertEqual(step.say[1:], [speaker_id.FAR_ONLY_INTRO, speaker_id.take_prompt("Quattro.")])

    def test_name_given_skips_the_question(self):
        step = self.en.start("anna")
        self.assertEqual(step.say, [speaker_id.INTRO, speaker_id.take_prompt("Uno.")])
        self.assertTrue(self.en.wants_voice)

    def test_bad_takes_are_asked_again(self):
        self.en.start("Anna")
        step = self.en.on_utterance("Uno", ANNA, 0.8)
        self.assertEqual(step.say, [speaker_id.TOO_SHORT, speaker_id.take_prompt("Uno.")])
        self.en.on_utterance("Uno.", ANNA, 3.0)
        step = self.en.on_utterance("Due.", BRUNO, 3.0)  # somebody else
        self.assertEqual(step.say, [speaker_id.UNCLEAR, speaker_id.take_prompt("Due.")])
        self.assertTrue(step.listen)

    def test_gives_up_after_too_many_bad_takes(self):
        self.en.start("Anna")
        for _ in range(speaker_id._MAX_BAD_TAKES - 1):
            self.assertTrue(self.en.on_utterance("eh", ANNA, 0.5).listen)
        self.assertEqual(self.en.on_utterance("eh", ANNA, 0.5),
                         speaker_id.Step([speaker_id.GIVE_UP], False))

    def test_name_not_understood(self):
        self.en.start()
        step = self.en.on_utterance("Non lo so davvero cosa dire", None, 2.0)
        self.assertEqual(step, speaker_id.Step([speaker_id.NAME_AGAIN], True))

    def test_cancel_by_voice_timeout_and_model_failure(self):
        self.en.start("Anna")
        self.assertEqual(self.en.on_utterance("Annulla.", None, 1.0),
                         speaker_id.Step([speaker_id.CANCELLED], False))
        self.assertEqual(self.en.on_utterance("Uno.", ANNA, 3.0), speaker_id.Step([], False))
        self.en.start("Anna")
        self.assertEqual(self.en.cancel("timeout"), speaker_id.Step([speaker_id.TIMEOUT], False))
        self.en.start("Anna")
        self.assertEqual(self.en.on_utterance("Uno.", None, 3.0),
                         speaker_id.Step([speaker_id.FAILED], False))
        self.assertEqual(speaker_id.SpeakerId(self.scfg).profiles, {})


class _Hid:
    def set_led(self, muted: bool) -> None:
        pass


class BridgeTest(_Tmp):
    def setUp(self):
        super().setUp()
        cfg = {
            "sample_rate": 16000, "chunk_size": 1024, "tts_sample_rate": 24000,
            "output_device": "null", "hid_mute_enabled": True, "idle_timeout_ms": 320,
            "activation": "wake_word", "wake_word": {"acks": [], "sleep_acks": []},
            "speaker_id": self.scfg,
        }
        self.stt = mock.Mock()
        self.b = VB.VoiceBridge(cfg, stt=self.stt, tts=mock.Mock(), hid=_Hid(),
                                deezer=mock.Mock())
        self.b._speaker_id.warm = lambda: None
        self.said: list = []
        self.b._enroll_say = self.said.extend
        self.b._say_goodbye = lambda _clip: None
        self.sent: list[str] = []
        self.b._run_text_turn = lambda gen, text, *a, **kw: self.sent.append(text)
        self.b._resume()
        self.b._pause_for_processing()  # an utterance was just committed

    def _turn(self, heard: str) -> None:
        self.stt.transcribe.return_value = heard
        gen = self.b._current_gen()
        if self.b._enroll.active:
            self.b._run_enroll_turn(gen, PCM_3S, 16000)
        else:
            self.b._run_turn(gen, PCM_3S, 16000, 0.0, {}, mock.Mock())

    def test_every_message_names_the_speaker(self):
        self._turn("Che ore sono?")
        self.b._speaker_id.save_profile("Anna", [ANNA])
        self.b._speaker_id.embed = lambda *_a: ANNA
        self._turn("Metti la musica.")
        self.b._speaker_id.embed = lambda *_a: BRUNO
        self._turn("Spegni la luce.")
        self.assertEqual(self.sent, ["[Sta parlando: Sconosciuto] Che ore sono?",
                                     "[Sta parlando: Anna] Metti la musica.",
                                     "[Sta parlando: Sconosciuto] Spegni la luce."])

    def test_idle_and_mute_end_the_conversation_memory(self):
        self.b._speaker_id.save_profile("Anna", [ANNA])
        for state in (S.IDLE_LISTENING, S.MUTED):
            self.b._speaker_id.embed = lambda *_a: ANNA
            self._turn("Metti la musica.")
            self.stt.transcribe.return_value = "Ok."
            self.b._speaker_id.embed = lambda *_a: BRUNO
            self.b._run_turn(self.b._current_gen(), PCM_1S, 16000, 0.0, {}, mock.Mock())
            self.b._set_state(S.PROCESSING, auto_idled=True)  # still the same conversation
            self.b._set_state(state)
            self.b._run_turn(self.b._current_gen(), PCM_1S, 16000, 0.0, {}, mock.Mock())
        self.assertEqual(self.sent, ["[Sta parlando: Anna] Metti la musica.",
                                     "[Sta parlando: Anna] Ok.",
                                     "[Sta parlando: Sconosciuto] Ok."] * 2)

    def test_far_request_starts_the_far_takes_of_who_asks(self):
        self.b._speaker_id.cfg["far_sentences"] = ["Quattro."]
        self.b._speaker_id.save_profile("Anna", [ANNA])
        self.b._speaker_id.embed = lambda *_a: ANNA
        self._turn("Impara la mia voce da lontano.")
        self.assertEqual(self.said, [speaker_id.FAR_ONLY_INTRO, speaker_id.take_prompt("Quattro.")])
        self.b._speaker_id.embed = lambda *_a: ANNA_FAR
        self._turn("Quattro.")
        self.assertEqual(self.said[-1], speaker_id.FAR_DONE)
        self.assertIn("Anna", self.b._speaker_id._far)
        self.assertEqual(self.sent, [])

    def test_wizard_runs_outside_the_agent(self):
        self.b._speaker_id.embed = lambda *_a: ANNA
        self._turn("Facciamo conoscenza.")
        self.assertEqual(self.said, [speaker_id.INTRO, speaker_id.ASK_NAME])
        self.assertIs(self.b._state, S.RECORDING)  # listening for the name
        for heard in ("Anna", "Uno.", "Due.", "Tre."):
            self.b._pause_for_processing()
            self.assertIs(self.b._state, S.PROCESSING)
            self._turn(heard)
            self.assertIs(self.b._state, S.RECORDING)
        self.assertEqual(self.said[-1], speaker_id.DONE)
        self.assertFalse(self.b._enroll.active)
        self.assertEqual(self.sent, [])  # nothing reached the agent
        self.b._pause_for_processing()
        self._turn("Metti la musica.")
        self.assertEqual(self.sent, ["[Sta parlando: Anna] Metti la musica."])

    def test_first_wake_without_a_known_voice_starts_the_wizard(self):
        self.b._resume()
        self.b._enter_idle("test")
        with mock.patch.object(VB.threading, "Thread") as thread:
            self.b._on_wake("ehi bainari")
        self.assertTrue(self.b._enroll.active)
        self.assertIs(thread.call_args.kwargs["target"].__func__, VB.VoiceBridge._enroll_step)
        self.b._enroll_step(*thread.call_args.kwargs["args"])
        self.assertEqual(self.said, [speaker_id.INTRO, speaker_id.ASK_NAME])
        self.assertIs(self.b._state, S.RECORDING)
        # Declined: the next wake is a normal one.
        self.b._pause_for_processing()
        self._turn("Annulla.")
        self.b._enter_idle("test")
        self.b._play_wake_ack = lambda _clip: None
        self.b._on_wake("ehi bainari")
        self.assertFalse(self.b._enroll.active)

    def test_wake_with_a_known_voice_is_a_normal_wake(self):
        self.b._speaker_id.save_profile("Anna", [ANNA])
        self.b._resume()
        self.b._enter_idle("test")
        self.b._play_wake_ack = lambda _clip: None
        self.b._on_wake("ehi bainari")
        self.assertIs(self.b._state, S.RECORDING)
        self.assertFalse(self.b._enroll.active)

    def test_hid_press_cancels_and_mutes(self):
        self._turn("Facciamo conoscenza.")
        gen = self.b._current_gen()
        self.b._on_hid_press()
        self.assertFalse(self.b._enroll.active)
        self.assertIs(self.b._state, S.MUTED)
        self.assertNotEqual(self.b._current_gen(), gen)
        self.b._listen_for_enrollment()  # the line that was playing ends
        self.assertIs(self.b._state, S.MUTED)

    def test_silence_ends_the_wizard(self):
        self._turn("Facciamo conoscenza.")
        del self.said[:]
        with mock.patch.object(VB.threading, "Thread") as thread:
            self.b._enter_idle("test")
        self.assertEqual(thread.call_args.kwargs["args"], ([speaker_id.TIMEOUT],))
        self.assertFalse(self.b._enroll.active)
        self.assertIs(self.b._state, S.IDLE_LISTENING)


if __name__ == "__main__":
    unittest.main(verbosity=2)
