# tests/conftest.py
"""
Session-level pytest configuration.

PyMuPDF (fitz) ships native C extensions (_extra.pyd, _mupdf.pyd …).
On this machine the DLL load fails, so we stub the entire package in
sys.modules before any test file imports backend.pdf_processor.
This lets patch("backend.pdf_processor.fitz") work correctly and
all other tests that don't touch PDF parsing remain unaffected.
"""
import sys
from unittest.mock import MagicMock

# ── Pre-mock fitz / pymupdf ────────────────────────────────────────────────
# Must happen before any module-level `import fitz` is executed.
_FITZ_MODS = (
    "fitz",
    "pymupdf",
    "pymupdf.extra",
    "pymupdf._extra",
    "pymupdf._mupdf",
)
for _mod in _FITZ_MODS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()