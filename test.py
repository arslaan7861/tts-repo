"""Diagnostic script: run this in Colab to see exactly what's on disk.

Usage in a Colab cell:
    !python /content/tts-repo/test.py
"""

from pathlib import Path

REPO_DIR = Path("/content/tts-repo")

print("=== repo dir contents ===")
if REPO_DIR.is_dir():
    for p in sorted(REPO_DIR.iterdir()):
        print(" ", p.name)
else:
    print(f"{REPO_DIR} does not exist")

print()
print("=== config.yaml ===")
config_path = REPO_DIR / "config.yaml"
if config_path.is_file():
    print(config_path.read_text())
else:
    print(f"{config_path} does not exist")

print()
print("=== voices/ ===")
voices_dir = REPO_DIR / "voices"
if voices_dir.is_dir():
    for child in sorted(voices_dir.iterdir()):
        print(" ", child, "->", sorted(p.name for p in child.iterdir()) if child.is_dir() else "")
else:
    print(f"{voices_dir} does not exist")

print()
print("=== charvoice import + resolved paths ===")
try:
    import sys

    sys.path.insert(0, str(REPO_DIR / "src"))
    from charvoice.config import load_config

    config = load_config(config_path)
    print("project_dir:", config.project_dir)
    print("resolve('voices'):", config.resolve("voices"))
    print("resolve('voices') exists:", config.resolve("voices").is_dir())
except Exception as exc:  # noqa: BLE001 -- diagnostic script, show everything
    print("charvoice import/load_config failed:", repr(exc))
