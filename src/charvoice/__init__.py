"""charvoice -- multi-character voiceover generator.

All logic lives here so that `main.ipynb` can stay a thin driver: every notebook
cell is a short call into this package. See requirements.md.

Nothing in this module imports torch, so the package is importable and testable
on a machine with no GPU.
"""

from __future__ import annotations

from charvoice.config import (
    AudioConfig,
    Config,
    EngineConfig,
    PathsConfig,
    TTSConfig,
    default_config,
    load_config,
)
from charvoice.errors import (
    AudioConversionError,
    AudioError,
    CharVoiceError,
    ConfigError,
    EngineError,
    GenerationError,
    GPUUnavailableError,
    MissingReferenceAudioError,
    ModelNotFoundError,
    ProfileError,
    ProfileNotFoundError,
    ScriptParseError,
    ScriptValidationError,
    ValidationIssue,
)
from charvoice.parser import ScriptLine, normalize_speaker, parse_script, parse_script_file
from charvoice.pipeline import (
    GenerationProgress,
    GenerationResult,
    generate_voiceover,
    validate_script,
)
from charvoice.profiles import ProfileStore, VoiceProfile, load_profiles

__version__ = "0.1.0"

__all__ = [
    # config
    "AudioConfig",
    "Config",
    "EngineConfig",
    "PathsConfig",
    "TTSConfig",
    "default_config",
    "load_config",
    # parser
    "ScriptLine",
    "normalize_speaker",
    "parse_script",
    "parse_script_file",
    # profiles
    "ProfileStore",
    "VoiceProfile",
    "load_profiles",
    # pipeline
    "GenerationProgress",
    "GenerationResult",
    "generate_voiceover",
    "validate_script",
    # errors
    "CharVoiceError",
    "ValidationIssue",
    "ScriptParseError",
    "ScriptValidationError",
    "ConfigError",
    "ProfileError",
    "ProfileNotFoundError",
    "MissingReferenceAudioError",
    "EngineError",
    "ModelNotFoundError",
    "GPUUnavailableError",
    "GenerationError",
    "AudioError",
    "AudioConversionError",
    "__version__",
]
