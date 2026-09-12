"""Command-line entry point.

Not a product feature -- this is what lets the full pipeline be run and
verified locally with the dummy engine, with no notebook involved. It calls
the exact same `generate_voiceover()` that `main.ipynb` does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from charvoice.config import load_config
from charvoice.errors import CharVoiceError
from charvoice.parser import parse_script_file
from charvoice.pipeline import GenerationProgress, generate_voiceover
from charvoice.profiles import load_profiles


def _print_progress(progress: GenerationProgress) -> None:
    tag = "cached" if progress.cached else "generating"
    print(f"[{progress.current}/{progress.total}] {tag}: {progress.speaker}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="charvoice", description=__doc__)
    parser.add_argument("script", type=Path, help="Path to the script text file.")
    parser.add_argument("-c", "--config", type=Path, default="config.yaml")
    parser.add_argument("-e", "--engine", default=None, help="Override tts.default_engine.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        profiles = load_profiles(
            config.resolve("voices"),
            config.paths.voices_combined and config.resolve("voices_combined"),
        )
        lines = parse_script_file(args.script)
        result = generate_voiceover(
            lines,
            profiles,
            config,
            engine_name=args.engine,
            output_path=args.output,
            overwrite=args.overwrite,
            progress_cb=_print_progress,
        )
    except CharVoiceError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
