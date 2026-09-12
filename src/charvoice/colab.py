"""Colab-only helpers: environment bootstrap, Drive mount, model fetch, download.

Every function here degrades gracefully outside Colab -- this module must
import cleanly with no `google.colab` package present, so the test suite can
run on a plain local machine (requirements.md section 2). Only *calling* a
function outside Colab is refused, and only where it would not make sense
(e.g. mounting a Drive that does not exist).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from charvoice.config import Config, EngineConfig

# Copied from the repo clone into the Drive project folder on first mount
# only -- see _seed_drive_project_dir. Model checkpoints are deliberately
# excluded: they are large and belong in Drive's own models/ once fetched
# there directly, not duplicated from the repo clone.
_SEED_ENTRIES = ("voices", "scripts", "config.yaml")


def in_colab() -> bool:
    """True if running inside a Google Colab runtime."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def setup_colab_environment(
    repo_url: str,
    repo_dir: Path | str = "/content/tts-repo",
    branch: str = "main",
    extras: str = "colab",
) -> Path:
    """Clone (or pull) the repo and install it editable. Idempotent.

    Re-running this cell must not re-clone or duplicate state: an existing
    checkout is `git pull`ed instead of re-cloned (requirements.md section 2).
    Returns the repo directory so later cells can reference it.
    """
    repo_dir = Path(repo_dir)

    if in_colab():
        try:
            _run(["nvidia-smi"])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(
                "WARNING: no GPU detected (nvidia-smi failed). "
                "Request a GPU runtime: Runtime > Change runtime type > GPU."
            )

    if (repo_dir / ".git").is_dir():
        _run(["git", "-C", str(repo_dir), "pull", "--ff-only"])
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--branch", branch, repo_url, str(repo_dir)])

    pip_target = f"{repo_dir}[{extras}]" if extras else str(repo_dir)
    _run([sys.executable, "-m", "pip", "install", "-e", pip_target, "-q"])

    if str(repo_dir / "src") not in sys.path:
        sys.path.insert(0, str(repo_dir / "src"))

    return repo_dir


def _seed_drive_project_dir(project_dir: Path, repo_dir: Path) -> None:
    """Copy voices/scripts/config.yaml into an empty Drive project folder.

    A freshly mounted Drive folder has none of this -- only the git clone
    does. Without seeding, `config.resolve("voices")` would silently point
    at an empty Drive directory and every profile lookup would report
    "no voice profile" even though the repo's profiles are right there
    (the bug this function exists to prevent). Only runs when the Drive
    folder doesn't already have its own `voices/`, so a user's edits there
    are never overwritten on a later mount.
    """
    if (project_dir / "voices").exists():
        return  # already seeded (or the user built their own) -- don't touch it

    print(f"First time using {project_dir}: copying starter files from the repo...")
    for name in _SEED_ENTRIES:
        src = repo_dir / name
        dest = project_dir / name
        if not src.exists() or dest.exists():
            continue
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copyfile(src, dest)
        print(f"  copied {name}")


def mount_drive(
    subpath: str = "character_voice_generator",
    repo_dir: Path | str | None = None,
) -> Path | None:
    """Mount Google Drive and return the project folder under MyDrive.

    Returns None outside Colab or if the user skips mounting -- callers treat
    a None return as "stay on the local checkout", never as an error
    (requirements.md section 13). On first use, seeds the Drive folder with
    the repo's voices/scripts/config.yaml so it starts as a working project
    directory rather than an empty one (pass `repo_dir` -- typically the
    clone from setup_colab_environment -- to enable this).
    """
    if not in_colab():
        print("mount_drive(): not running in Colab, skipping.")
        return None

    from google.colab import drive  # noqa: PLC0415

    drive.mount("/content/drive")
    project_dir = Path("/content/drive/MyDrive") / subpath
    project_dir.mkdir(parents=True, exist_ok=True)

    if repo_dir is not None:
        _seed_drive_project_dir(project_dir, Path(repo_dir))

    return project_dir


_GPT_SOVITS_REPO_URL = "https://github.com/RVC-Boss/GPT-SoVITS.git"
_GPT_SOVITS_WEIGHTS_URL = "https://huggingface.co/lj1995/GPT-SoVITS"
# Marker files checked per source before (re-)fetching it, so a re-run of
# this cell is a no-op once both are present -- cheap existence checks, no
# hashing, matching requirements.md section 2's "no repeated downloads" rule.
_GPT_SOVITS_PACKAGE_MARKER = "GPT_SoVITS/TTS_infer_pack/TTS.py"
_GPT_SOVITS_WEIGHTS_MARKER = "gsv-v2final-pretrained/s2G2333k.pth"


def _fetch_gpt_sovits(model_dir: Path) -> None:
    """Clone the GPT-SoVITS repo (for its Python package) and pull v2 weights.

    Two independent sources land in the same `model_dir`: the upstream repo
    (needed on sys.path for `from GPT_SoVITS.TTS_infer_pack.TTS import TTS`)
    and the pretrained checkpoints from HuggingFace (too large to vendor into
    this repo, per requirements.md section 14's "don't hard-code model paths"
    and section 18's model-size guidance). Each is skipped independently if
    already present, so interrupting one doesn't force re-fetching the other.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    if (model_dir / _GPT_SOVITS_PACKAGE_MARKER).is_file():
        print(f"fetch_model(): GPT-SoVITS package already present under {model_dir}.")
    else:
        print(f"fetch_model(): cloning GPT-SoVITS into {model_dir} ...")
        _run(["git", "clone", "--depth", "1", _GPT_SOVITS_REPO_URL, str(model_dir)])

    if (model_dir / _GPT_SOVITS_WEIGHTS_MARKER).is_file():
        print(f"fetch_model(): pretrained v2 weights already present under {model_dir}.")
    else:
        print(f"fetch_model(): cloning pretrained weights into {model_dir} ...")
        # The HuggingFace repo is itself a git repo (with git-lfs for the
        # large checkpoint files); clone it directly rather than adding a
        # huggingface_hub dependency for a single one-time download.
        _run(["git", "lfs", "install", "--skip-repo"])
        _run(["git", "clone", _GPT_SOVITS_WEIGHTS_URL, str(model_dir / "_weights_tmp")])
        weights_tmp = model_dir / "_weights_tmp"
        for child in weights_tmp.iterdir():
            if child.name == ".git":
                continue
            child.rename(model_dir / child.name)
        shutil.rmtree(weights_tmp, ignore_errors=True)

    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))


def fetch_model(engine: str, engine_config: EngineConfig) -> Path | None:
    """Ensure the engine's model files are present, downloading if needed.

    Re-running this cell is a no-op once the model is present -- checked via
    cheap marker-file existence, not by re-downloading and diffing
    (requirements.md section 2, section 19).
    """
    if not engine_config.model_dir:
        print(f"fetch_model(): engine {engine!r} has no model_dir configured, nothing to fetch.")
        return None

    model_dir = Path(engine_config.model_dir).expanduser()

    if engine == "gpt-sovits":
        _fetch_gpt_sovits(model_dir)
        return model_dir

    if model_dir.is_dir() and any(model_dir.iterdir()):
        print(f"fetch_model(): {model_dir} already populated, skipping.")
        return model_dir

    print(
        f"fetch_model(): {model_dir} is empty or missing. Populate it with the "
        f"{engine!r} checkpoint files before generating (requirements.md section 14)."
    )
    return model_dir


def offer_download(path: Path | str) -> None:
    """Hand the final file to the browser for download. No-op outside Colab."""
    path = Path(path)
    if not in_colab():
        print(f"offer_download(): not running in Colab. File is at: {path}")
        return

    from google.colab import files  # noqa: PLC0415

    files.download(str(path))


def project_dir_for(config: Config, drive_dir: Path | None) -> Config:
    """Re-base `config` onto the mounted Drive folder, if one was mounted.

    Threading this explicitly through every call site is what makes Drive
    mounting actually take effect -- leaving paths resolved against `cwd()`
    after mounting Drive is a documented trap (requirements.md section 2):
    the profiles and output would silently keep reading/writing the local
    checkout instead of Drive.
    """
    if drive_dir is None:
        return config
    return config.with_project_dir(drive_dir)
