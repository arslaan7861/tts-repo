"""Engine registration and the process-level loaded-model cache.

Re-running a Colab generate cell must reuse an already-loaded model rather
than reload it (requirements.md section 19). `get_engine` is keyed on engine
name plus `model_fingerprint(config)`, so the same engine+config always
returns the same instance, while a changed config (e.g. a different
`model_dir`) produces a fresh one -- the caller decides whether/when to
`unload()` the old one.

CRITICAL: this module must not import `dummy` or `gpt_sovits` at module
scope. `gpt_sovits` may eventually need torch, and this package must stay
importable with no torch installed. The built-in engines are imported lazily,
on first use, each behind its own try/except so one adapter's missing deps
never hide another's availability.
"""

from __future__ import annotations

from collections.abc import Callable

from charvoice.config import Config, EngineConfig
from charvoice.engines.base import Engine
from charvoice.errors import EngineNotFoundError

_REGISTRY: dict[str, type[Engine]] = {}
_INSTANCE_CACHE: dict[tuple[str, str], Engine] = {}
_builtins_loaded = False


def register(name: str) -> Callable[[type[Engine]], type[Engine]]:
    """Class decorator. Registers `cls` under `name`."""

    def decorator(cls: type[Engine]) -> type[Engine]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def _load_builtin_engines() -> None:
    """Import the built-in adapter modules once, tolerating missing deps.

    Each import is isolated so that, e.g., gpt_sovits failing to import (its
    own optional deps absent) never prevents dummy from registering.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True

    try:
        import charvoice.engines.dummy  # noqa: F401
    except ImportError:
        pass

    try:
        import charvoice.engines.gpt_sovits  # noqa: F401
    except ImportError:
        pass


def available_engines() -> list[str]:
    """Names of every registered engine."""
    _load_builtin_engines()
    return sorted(_REGISTRY)


def get_engine(name: str, config: Config) -> Engine:
    """Return the (possibly cached) loaded engine instance for `name`.

    Process-level cache keyed by (name, model_fingerprint). Calling again
    with the same engine+config returns the SAME instance -- re-running a
    notebook cell must never reload the model. A different config produces a
    new cached instance; the old one is NOT auto-unloaded.
    """
    _load_builtin_engines()

    cls = _REGISTRY.get(name)
    if cls is None:
        available = ", ".join(available_engines()) or "(none)"
        raise EngineNotFoundError(f'No engine registered as "{name}". Available: {available}')

    engine_config: EngineConfig = config.tts.engine(name)

    # Fingerprinting needs an instance, but we must not create/load a second
    # real one just to compute it -- an unloaded throwaway instance is cheap
    # (model_fingerprint only touches engine_config / checkpoint files).
    probe = cls()
    fingerprint = probe.model_fingerprint(engine_config)
    cache_key = (name, fingerprint)

    cached = _INSTANCE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    engine = cls()
    engine.load(engine_config)
    _INSTANCE_CACHE[cache_key] = engine
    return engine


def clear_engine_cache() -> None:
    """Drop every cached engine instance. Mainly for tests."""
    _INSTANCE_CACHE.clear()
