"""Exception hierarchy for charvoice.

Design rule carried through the whole package: validation collects *every*
problem and raises once, so the user fixes all of them in one pass instead of
re-running to discover the next error. See requirements.md section 11.
"""

from __future__ import annotations

from dataclasses import dataclass


class CharVoiceError(Exception):
    """Base class for every error this package raises."""


# --------------------------------------------------------------------------
# Validation issues
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a script or profile set.

    `message` is already user-facing and complete. `hint` is an optional
    follow-up action, rendered under a "Create:" / "Fix:" heading.
    """

    kind: str
    message: str
    line_no: int | None = None
    speaker: str | None = None
    hint: str | None = None

    def render(self) -> str:
        block = f"ERROR:\n{self.message}"
        if self.hint:
            block += f"\n\n{self.hint}"
        return block


class IssueListError(CharVoiceError):
    """Base for errors that carry a list of collected issues."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = list(issues)
        super().__init__(self._render())

    def _render(self) -> str:
        return "\n\n".join(issue.render() for issue in self.issues)

    def __str__(self) -> str:
        return self._render()


class ScriptParseError(IssueListError):
    """The script text could not be parsed into speaker/line pairs."""


class ScriptValidationError(IssueListError):
    """The script parsed, but references speakers or data that do not resolve."""


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


class ConfigError(CharVoiceError):
    """config.yaml is missing, malformed, or internally inconsistent."""


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


class ProfileError(CharVoiceError):
    """Base for voice-profile problems."""


class ProfileLoadError(ProfileError):
    """A profile file exists but could not be read or understood."""


class ProfileNotFoundError(ProfileError):
    """A speaker was requested that has no profile."""


class MissingReferenceAudioError(ProfileError):
    """A profile names reference audio that is not on disk."""


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------


class EngineError(CharVoiceError):
    """Base for TTS engine problems."""


class EngineNotFoundError(EngineError):
    """No engine is registered under the requested name."""


class ModelNotFoundError(EngineError):
    """The engine loaded but its model files are absent."""


class GPUUnavailableError(EngineError):
    """The engine requires a GPU and none is available."""


class GenerationError(EngineError):
    """The engine failed while synthesising a line."""


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------


class AudioError(CharVoiceError):
    """Base for audio processing problems."""


class AudioConversionError(AudioError):
    """Format conversion failed, typically a missing or failing ffmpeg."""
