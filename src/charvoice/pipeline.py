"""Validation and generation orchestration -- the glue between every other module.

This is the only module that imports `config`, `parser`, `profiles`, `cache`,
`audio` and `engines` together. Everything else stays decoupled so those
pieces can be developed, tested, and swapped independently (requirements.md
section 15).

Two rules shape every function here:

1. **Validate everything before spending any GPU time.** Parsing and profile
   checks are cheap; loading a model is not. `validate_script` always runs,
   and raises with every problem collected, before `generate_voiceover` ever
   calls `get_engine`.
2. **Speaker grouping is for compute efficiency only; output order is the
   script order.** Lines are grouped by speaker so `Engine.prepare_speaker`
   runs once per character, but results are keyed by each line's original
   `index` and walked back in script order before being written out.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from charvoice.audio import (
    AudioSegment,
    StreamingWavWriter,
    conform,
    export_mp3,
    load_segment,
    save_segment,
    silence,
)
from charvoice.cache import CacheKey, SegmentCache
from charvoice.config import Config
from charvoice.engines.base import Engine, SpeakerHandle
from charvoice.engines.registry import get_engine
from charvoice.errors import ScriptValidationError, ValidationIssue
from charvoice.parser import ScriptLine, speakers_in
from charvoice.profiles import ProfileStore, fingerprint_profile


def validate_script(
    lines: list[ScriptLine], profiles: ProfileStore, engine: Engine | None = None
) -> None:
    """Check every line against `profiles`, collecting every problem.

    Raises ScriptValidationError listing ALL unknown speakers and, when
    `engine` is given, all profiles missing a field that engine requires
    (requirements.md section 11) -- never just the first one found, so a
    script can be fixed in one pass instead of a rerun-per-error loop.
    """
    issues: list[ValidationIssue] = []
    seen_unknown: set[str] = set()

    for line in lines:
        if line.speaker_id in profiles or line.speaker_id in seen_unknown:
            continue
        if line.speaker_id not in profiles:
            seen_unknown.add(line.speaker_id)
            issues.append(
                ValidationIssue(
                    kind="unknown_speaker",
                    message=(
                        f'Character "{line.speaker_raw}" appears in the script '
                        "but has no voice profile."
                    ),
                    speaker=line.speaker_id,
                    line_no=line.line_no,
                    hint=f"Create:\nvoices/{line.speaker_id}/profile.yaml",
                )
            )

    if engine is not None:
        required = engine.required_profile_fields()
        if required:
            for speaker_id in speakers_in(lines):
                if speaker_id not in profiles:
                    continue  # already reported above
                profile = profiles.get(speaker_id)
                missing = sorted(f for f in required if getattr(profile, f, None) in (None, ""))
                if missing:
                    issues.append(
                        ValidationIssue(
                            kind="missing_profile_field",
                            message=(
                                f'Voice profile "{speaker_id}" is missing field(s) required '
                                f'by engine "{profile.engine}": {", ".join(missing)}.'
                            ),
                            speaker=speaker_id,
                            hint=f"Edit: voices/{speaker_id}/profile.yaml",
                        )
                    )

    if issues:
        raise ScriptValidationError(issues)


@dataclass
class GenerationProgress:
    current: int
    total: int
    speaker: str
    cached: bool = False


@dataclass
class GenerationResult:
    output_path: Path
    characters: list[str]
    line_count: int
    duration_s: float
    cache_hits: int
    cache_misses: int

    def summary(self) -> str:
        """Render the exact block shown in requirements.md section 12, Cell 8."""
        chars = "\n".join(f"- {name}" for name in self.characters)
        return (
            "Generation complete.\n\n"
            f"Characters detected:\n{chars}\n\n"
            f"Lines:\n{self.line_count}\n\n"
            f"Output:\n{self.output_path}"
        )


def _group_by_speaker(lines: list[ScriptLine]) -> dict[str, list[ScriptLine]]:
    """Group lines by speaker, preserving first-appearance order of speakers."""
    groups: dict[str, list[ScriptLine]] = {}
    for line in lines:
        groups.setdefault(line.speaker_id, []).append(line)
    return groups


def _resolve_segments(
    lines: list[ScriptLine],
    profiles: ProfileStore,
    engine: Engine,
    engine_name: str,
    model_fingerprint: str,
    cache: SegmentCache,
    progress_cb: Callable[[GenerationProgress], None] | None,
) -> dict[int, Path]:
    """Resolve every line to a path holding its raw (unconformed) synthesized audio.

    Cache is checked per line *before* any speaker grouping happens, so a
    character whose lines are all cache hits never triggers
    `engine.prepare_speaker()` -- that setup is skipped entirely for them.
    """
    results: dict[int, Path] = {}
    pending: list[ScriptLine] = []
    keys: dict[int, CacheKey] = {}

    for line in lines:
        profile = profiles.get(line.speaker_id)
        key = CacheKey(
            text=line.text,
            speaker_id=line.speaker_id,
            engine_name=engine_name,
            profile_fingerprint=fingerprint_profile(profile),
            model_fingerprint=model_fingerprint,
        )
        keys[line.index] = key
        cached_path = cache.get(key)
        if cached_path is not None:
            results[line.index] = cached_path
        else:
            pending.append(line)

    total = len(lines)
    done = 0
    for line in lines:
        if line.index in results:
            done += 1
            if progress_cb is not None:
                progress_cb(GenerationProgress(done, total, line.speaker_raw, cached=True))

    for speaker_id, speaker_lines in _group_by_speaker(pending).items():
        profile = profiles.get(speaker_id)
        handle: SpeakerHandle = engine.prepare_speaker(profile)
        texts = [line.text for line in speaker_lines]
        segments: list[AudioSegment] = engine.generate_batch(texts, handle)

        for line, segment in zip(speaker_lines, segments, strict=True):
            raw_path = cache.path_for(keys[line.index]).with_suffix(".raw.wav")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            save_segment(segment, raw_path)
            results[line.index] = cache.put(keys[line.index], raw_path)
            raw_path.unlink(missing_ok=True)

            done += 1
            if progress_cb is not None:
                progress_cb(GenerationProgress(done, total, line.speaker_raw, cached=False))

    return results


def _ordered_segments(
    lines: list[ScriptLine], segment_paths: dict[int, Path]
) -> Iterator[tuple[ScriptLine, AudioSegment]]:
    """Yield (line, segment) pairs in script order, loading each just-in-time.

    A generator rather than a list: the caller streams straight to disk, so
    peak memory is one segment regardless of script length.
    """
    for line in lines:
        yield line, load_segment(segment_paths[line.index])


def generate_voiceover(
    lines: list[ScriptLine],
    profiles: ProfileStore,
    config: Config,
    engine_name: str | None = None,
    output_path: Path | str | None = None,
    overwrite: bool = False,
    progress_cb: Callable[[GenerationProgress], None] | None = None,
) -> GenerationResult:
    """Validate, synthesize, and assemble one combined voiceover file.

    Raises ScriptValidationError before loading any model if validation
    fails. Output is staged under `paths.temp` and only renamed into place on
    success, so a failed or interrupted run never corrupts a previous output
    (requirements.md section 8).
    """
    engine_name = engine_name or config.tts.default_engine
    engine = get_engine(engine_name, config)

    validate_script(lines, profiles, engine=engine)
    if not lines:
        raise ScriptValidationError(
            [ValidationIssue(kind="empty_script", message="The script has no spoken lines.")]
        )

    output_path = Path(output_path) if output_path else config.resolve("output") / "voiceover.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass overwrite=True to replace it."
        )

    temp_dir = config.resolve("temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    staging_path = temp_dir / f".{output_path.name}.staging.wav"

    cache_dir = config.resolve("cache")
    cache = SegmentCache(cache_dir)

    model_fp = engine.model_fingerprint(config.tts.engine(engine_name))
    segment_paths = _resolve_segments(
        lines, profiles, engine, engine_name, model_fp, cache, progress_cb
    )

    with StreamingWavWriter(
        staging_path, config.audio.sample_rate, config.audio.channels, config.audio.subtype
    ) as writer:
        prev_line: ScriptLine | None = None
        for line, raw_segment in _ordered_segments(lines, segment_paths):
            if prev_line is not None:
                pause_ms = config.audio.pause_between_lines_ms
                after_narrator_ms = config.audio.pause_after_narrator_ms
                if prev_line.speaker_id == "narrator" and after_narrator_ms is not None:
                    pause_ms = after_narrator_ms
                writer.write(silence(pause_ms, config.audio.sample_rate, config.audio.channels))
            writer.write(conform(raw_segment, config.audio))
            prev_line = line
    duration_s = writer.frames_written / float(config.audio.sample_rate)

    staging_path.replace(output_path)

    if config.audio.output_format == "mp3":
        export_mp3(output_path, output_path.with_suffix(".mp3"))

    return GenerationResult(
        output_path=output_path,
        characters=[lines[i].speaker_raw for i in _first_seen_indices(lines)],
        line_count=len(lines),
        duration_s=duration_s,
        cache_hits=cache.hits,
        cache_misses=cache.misses,
    )


def _first_seen_indices(lines: list[ScriptLine]) -> list[int]:
    """Index of each speaker's first line, in first-appearance order."""
    seen: dict[str, int] = {}
    for line in lines:
        seen.setdefault(line.speaker_id, line.index)
    return list(seen.values())
