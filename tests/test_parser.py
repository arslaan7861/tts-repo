from __future__ import annotations

import pytest

from charvoice.errors import ScriptParseError
from charvoice.parser import ScriptLine, normalize_speaker, parse_script, speakers_in


def test_basic_two_speakers():
    text = "NARRATOR: The city was silent.\n\nSPIDER-MAN: Something feels wrong."
    lines = parse_script(text)
    assert [line.speaker_id for line in lines] == ["narrator", "spider-man"]
    assert lines[0].text == "The city was silent."
    assert lines[1].text == "Something feels wrong."


def test_blank_lines_ignored():
    text = "NARRATOR: One.\n\n\n\nMILES: Two.\n"
    lines = parse_script(text)
    assert len(lines) == 2


def test_order_preserved():
    text = "\n".join(f"A: line {i}" for i in range(10))
    lines = parse_script(text)
    assert [line.index for line in lines] == list(range(10))
    assert [line.text for line in lines] == [f"line {i}" for i in range(10)]


def test_case_insensitive_speaker_matching():
    lines = parse_script("Spider-Man: Hi.\nspider-man: Bye.\nSPIDER-MAN: Again.")
    assert {line.speaker_id for line in lines} == {"spider-man"}


def test_speaker_raw_preserves_original_case():
    lines = parse_script("Spider-Man: Hi.")
    assert lines[0].speaker_raw == "Spider-Man"
    assert lines[0].speaker_id == "spider-man"


def test_text_after_first_colon_only():
    lines = parse_script('NARRATOR: He said: "go now."')
    assert lines[0].text == 'He said: "go now."'


def test_missing_colon_raises():
    with pytest.raises(ScriptParseError) as exc_info:
        parse_script("This has no colon at all")
    assert exc_info.value.issues[0].kind == "missing_speaker"


def test_empty_spoken_line_raises():
    with pytest.raises(ScriptParseError) as exc_info:
        parse_script("NARRATOR:   ")
    assert exc_info.value.issues[0].kind == "empty_line"


def test_multiple_problems_collected_together():
    text = "\n".join(
        [
            "NARRATOR: fine line",
            "no colon here",
            "MILES:   ",
            "This is way too many words to be a speaker name: so it is rejected",
        ]
    )
    with pytest.raises(ScriptParseError) as exc_info:
        parse_script(text)
    issues = exc_info.value.issues
    assert len(issues) == 3
    kinds = {issue.kind for issue in issues}
    assert "missing_speaker" in kinds
    assert "empty_line" in kinds
    assert "invalid_speaker" in kinds


def test_error_message_includes_line_number():
    with pytest.raises(ScriptParseError) as exc_info:
        parse_script("NARRATOR: fine\nno colon here")
    assert "Line 2" in str(exc_info.value)


def test_valid_script_does_not_raise():
    lines = parse_script("NARRATOR: a.\nMILES: b.\nSPIDER-MAN: c.")
    assert len(lines) == 3


def test_normalize_speaker_collapses_whitespace_and_case():
    assert normalize_speaker("  Spider-Man ") == "spider-man"
    assert normalize_speaker("Aunt May") == "aunt-may"
    assert normalize_speaker("MILES") == "miles"


def test_normalize_speaker_collapses_repeated_hyphens():
    assert normalize_speaker("Spider---Man") == "spider-man"


def test_speakers_in_first_appearance_order():
    lines = parse_script("NARRATOR: a.\nMILES: b.\nNARRATOR: c.\nSPIDER-MAN: d.")
    assert speakers_in(lines) == ["narrator", "miles", "spider-man"]


def test_unicode_speaker_name():
    lines = parse_script("José: hola")
    assert lines[0].speaker_id == "josé"


def test_line_no_is_one_based_source_line():
    lines = parse_script("\n\nNARRATOR: third line")
    assert lines[0].line_no == 3


def test_scriptline_is_frozen_and_hashable():
    line = ScriptLine(index=0, speaker_raw="A", speaker_id="a", text="hi", line_no=1)
    with pytest.raises(AttributeError):
        line.text = "changed"  # type: ignore[misc]
