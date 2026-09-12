"""Colab-only helpers: environment bootstrap, Drive mount, model fetch, download.

Every function here degrades gracefully outside Colab -- this module must
import cleanly with no `google.colab` package present, so the test suite can
run on a plain local machine (requirements.md section 2). Only *calling* a
function outside Colab is refused, and only where it would not make sense
(e.g. mounting a Drive that does not exist).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from charvoice.config import Config, EngineConfig


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


def mount_drive(subpath: str = "character_voice_generator") -> Path | None:
    """Mount Google Drive and return the project folder under MyDrive.

    Returns None outside Colab or if the user skips mounting -- callers treat
    a None return as "stay on the local checkout", never as an error
    (requirements.md section 13).
    """
    if not in_colab():
        print("mount_drive(): not running in Colab, skipping.")
        return None

    from google.colab import drive  # noqa: PLC0415

    drive.mount("/content/drive")
    project_dir = Path("/content/drive/MyDrive") / subpath
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir


def fetch_model(engine: str, engine_config: EngineConfig) -> Path | None:
    """Ensure the engine's model files are present, downloading if needed.

    Currently a structural hook: real per-engine download logic (cloning
    GPT-SoVITS, pulling checkpoints) belongs here once an engine needs it.
    Re-running this cell must be a no-op when the model is already present
    (requirements.md section 2) -- checked via `model_dir` already existing
    and non-empty.
    """
    if not engine_config.model_dir:
        print(f"fetch_model(): engine {engine!r} has no model_dir configured, nothing to fetch.")
        return None

    model_dir = Path(engine_config.model_dir).expanduser()
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
