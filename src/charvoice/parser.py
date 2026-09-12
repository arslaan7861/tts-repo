"""Script parsing: raw text -> ordered ScriptLine list.

Input format (requirements.md section 3):

    NARRATOR: The city was completely silent that night.

    SPIDER-MAN: Something feels wrong.

Only the *first* colon splits speaker from text, so dialogue may contain colons.
Parsing collects every malformed line and raises once, so a script with several
problems reports all of them together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from charvoice.errors import ScriptParseError, ValidationIssue

# A speaker name is short and has no sentence punctuation. This guard is what
# stops a prose line containing a colon ("He said: go") from being mistaken for
# a speaker cue.
_MAX_SPEAKER_WORDS = 5
_SPEAKER_RE = re.compile(r"^[\w][\w '\-.&]*$", re.UNICODE)


@dataclass(frozen=True)
class ScriptLine:
    index: int
    """0-based position among spoken lines. This is the ordering key."""

    speaker_raw: str
    """The name as written, e.g. "SPIDER-MAN". Used in error messages."""

    speaker_id: str
    """Normalised id used to look up a profile, e.g. "spider-man"."""

    text: str
    line_no: int
    """1-based source line number, for error messages."""


def normalize_speaker(name: str) -> str:
    """Normalise a speaker name to a stable profile id.

    Case-insensitive, whitespace collapsed to single hyphens:
    "  Spider-Man " -> "spider-man", "Aunt May" -> "aunt-may".
    """
    collapsed = "-".join(name.strip().lower().split())
    return re.sub(r"-{2,}", "-", collapsed).strip("-")


def looks_like_speaker(candidate: str) -> bool:
    """True if the text before a colon plausibly names a speaker."""
    candidate = candidate.strip()
    if not candidate or len(candidate.split()) > _MAX_SPEAKER_WORDS:
        return False
    return bool(_SPEAKER_RE.match(candidate))


def parse_script(raw_text: str) -> list[ScriptLine]:
    """Parse script text into ordered lines.

    Raises ScriptParseError listing every problem found.
    """
    lines: list[ScriptLine] = []
    issues: list[ValidationIssue] = []

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        speaker_part, sep, text_part = stripped.partition(":")
        if not sep:
            issues.append(
                ValidationIssue(
                    kind="missing_speaker",
                    message=(
                        f'Line {line_no} has no speaker. Expected "NAME: spoken text".\n'
                        f"  {stripped}"
                    ),
                    line_no=line_no,
                )
            )
            continue

        if not looks_like_speaker(speaker_part):
            issues.append(
                ValidationIssue(
                    kind="invalid_speaker",
                    message=(
                        f"Line {line_no} does not start with a speaker name.\n"
                        f"  {stripped}\n"
                        f'Expected "NAME: spoken text".'
                    ),
                    line_no=line_no,
                )
            )
            continue

        text = text_part.strip()
        if not text:
            issues.append(
                ValidationIssue(
                    kind="empty_line",
                    message=(
                        f"Line {line_no} has no spoken text after "
                        f'"{speaker_part.strip()}:".'
                    ),
                    line_no=line_no,
                    speaker=speaker_part.strip(),
                )
            )
            continue

        lines.append(
            ScriptLine(
                index=len(lines),
                speaker_raw=speaker_part.strip(),
                speaker_id=normalize_speaker(speaker_part),
                text=text,
                line_no=line_no,
            )
        )

    if issues:
        raise ScriptParseError(issues)
    return lines


def parse_script_file(path) -> list[ScriptLine]:
    """Parse a script from a file path."""
    from pathlib import Path

    return parse_script(Path(path).expanduser().read_text())


def speakers_in(lines: list[ScriptLine]) -> list[str]:
    """Distinct speaker ids in first-appearance order."""
    seen: dict[str, None] = {}
    for line in lines:
        seen.setdefault(line.speaker_id, None)
    return list(seen)
