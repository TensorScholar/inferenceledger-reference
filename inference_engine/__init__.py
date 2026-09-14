"""Development import shim for the src-layout package.

The installable package lives in `src/inference_engine`. This shim keeps
`python -c "import inference_engine"` working from a fresh checkout before an
editable install has been performed.
"""

from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "inference_engine"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]

__all__: list[str] = []

