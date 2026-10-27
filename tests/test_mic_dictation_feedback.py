"""Dictation feedback inside the mic button (state, level, silence auto-stop).

The behaviour lives in `static/boot.js`, which this suite cannot execute, so
these are contract checks on the published surface: the two child elements the
button gains, the CSS that renders them, the lifecycle hooks that drive them,
and the one preference that changes user-visible behaviour. They fail if the
wiring is removed, renamed, or driven by polling instead of the capture's own
events.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOT_JS = (ROOT / "static" / "boot.js").read_text(encoding="utf-8")
STYLE_CSS = (ROOT / "static" / "style.css").read_text(encoding="utf-8")


def test_button_gets_state_and_level_children():
    assert "micStateEl.className='mic-state'" in BOOT_JS
    assert "micLevelEl.className='mic-level'" in BOOT_JS
    assert "btn.append(micStateEl,micLevelEl)" in BOOT_JS


def test_children_are_decorative_for_screen_readers():
    # The timer and the level bar duplicate the tooltip state; exposing them
    # would make every second of a dictation an announcement.
    assert BOOT_JS.count("micStateEl.setAttribute('aria-hidden','true')") == 1
    assert BOOT_JS.count("micLevelEl.setAttribute('aria-hidden','true')") == 1


def test_state_is_driven_by_the_capture_lifecycle():
    # rec starts with the capture, thinking with the request, ready/empty when
    # the transcript is committed. No polling: the DOM is written from one place.
    assert "_micPaint('rec')" in BOOT_JS
    assert "_micPaint('think','···')" in BOOT_JS
    assert "_micPaint('ready','✓ '+_micWordCount(clean)+'p',_MIC_READY_MS)" in BOOT_JS
    assert "_micPaint('empty','0p',_MIC_EMPTY_MS)" in BOOT_JS


def test_level_meter_uses_the_capture_stream_not_a_second_capture():
    # A second getUserMedia would prompt a second permission and open two input
    # streams for one dictation.
    assert "_micStartMeter(captureStream)" in BOOT_JS
    meter = BOOT_JS.split("function _micStartMeter(stream){", 1)[1].split("function _micWordCount", 1)[0]
    assert "getUserMedia" not in meter
    assert "createMediaStreamSource(stream)" in meter


def test_silence_auto_stop_is_opt_out_and_bounded():
    assert "_MIC_AUTOSTOP_KEY='mic_autostop'" in BOOT_JS
    assert "localStorage.getItem(_MIC_AUTOSTOP_KEY)!=='0'" in BOOT_JS
    # It may only fire after speech was actually heard, never on a quiet room.
    assert "_MIC_SILENCE_MS=1600" in BOOT_JS
    auto_stop = BOOT_JS.split("function _micStartMeter(stream){", 1)[1].split("function _micWordCount", 1)[0]
    assert "if(!_micSpokeAt) return;" in auto_stop
    assert "_micAutoStopEnabled()" in auto_stop


def test_css_renders_state_inside_the_button_without_new_i18n_keys():
    assert ".mic-btn .mic-state{display:none;" in STYLE_CSS
    assert ".mic-btn.mic-busy>svg{display:none;}" in STYLE_CSS
    assert ".mic-btn.mic-busy .mic-state{display:inline-block;" in STYLE_CSS
    assert ".mic-btn .mic-level{position:absolute;" in STYLE_CSS
    assert "@media(prefers-reduced-motion:reduce){.mic-btn .mic-level{transition:none;}}" in STYLE_CSS


def test_tooltip_anchor_is_untouched():
    # .has-tooltip is what positions the button; adding absolutely positioned
    # children must not introduce a competing position rule on .mic-btn.
    assert ".has-tooltip{position:relative;}" in STYLE_CSS
    assert ".mic-btn{position:" not in STYLE_CSS
