"""TTS engine adapters.

Deliberately empty: importing this package must never import `dummy` or
`gpt_sovits` eagerly. `registry.get_engine()` imports them lazily so that
`gpt_sovits`'s (eventual) torch dependency never blocks importing charvoice
on a machine with no GPU. See requirements.md sections 2 and 15.
"""

from __future__ import annotations
