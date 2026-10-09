# TODO

## Replace the state Events with an explicit state enum

Left out of the 2026-10-10 resource/complexity review because it needs a
plan and a test run on the real Jabra.

**Problem.** The bridge's mode is spread over four `threading.Event`s —
`recording`, `_wake_armed`, `_processing`, `_auto_idled` — read and written
in combination by seven `@_transition` methods (`_on_hid_press`, `_resume`,
`_enter_idle`, `_pause_for_processing`, `_resume_after_processing`,
`_unidle_for_reply`, `_unmute_for_external`, plus `_disable_wake`).
`_on_hid_press` alone has five branches over four flags. Every transition
must also remember to keep the LED / firmware mute (`hid.set_led`) in step.

**Proposal.**

- One enum: `RECORDING`, `PROCESSING`, `IDLE_LISTENING` (wake armed),
  `MUTED`, plus the `auto_idled` bit (resume the mic when the reply starts).
- One `_set_state(new)` under `_state_lock` that derives everything else:
  `recording` (still an Event — the recorder blocks on it), the wake routing,
  and the `set_led` write. Transitions only choose the next state.
- Write the transition table in AGENTS.md (state × event → state) and pin it
  with one table-driven test.

**Constraints to keep.**

- HID press while processing is a privacy mute, not a cancel: no gen bump,
  the reply still plays and the mic stays muted afterwards.
- Idle-but-listening must leave the firmware mic unmuted (LED off); red
  means privacy mute only.
- Player auto-resume fires only after an *auto* idle, never after an
  explicit HID mute.
- Transitions never block while holding `_state_lock`.

**Cost / risk.** About 80 test lines set or assert those Events directly
(`tests/test_voice_bridge_endpointer.py`, `test_wake_word.py`,
`test_ux_improvements.py`); they have to move to the new state API. Verify
on the device afterwards with `tests/test_hid_interactive.py` and a few
wake → reply → idle → HID mute cycles.
