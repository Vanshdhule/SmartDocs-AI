# backend/text_cleaner.py
"""
Text Cleaning Module
────────────────────
Permissive cleaning — only removes what genuinely hurts.
Preserves technical content: numbers, citations, correlations, symbols.
"""

import re
import unicodedata
from typing import Optional


class TextCleaner:

    def __init__(self):
        pass

    def remove_extra_whitespace(self, text: str) -> str:
        """Collapse multiple spaces, tabs, and newlines into single spaces."""
        if not text or not text.strip():
            return ""
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def remove_headers_footers(self, text: str) -> str:
        """
        Remove recurring header/footer patterns only:
        - Explicit 'Page N' / 'Page N of M'
        - Lines that are ONLY a bare page number

        Numbers inside sentences (years, stats, CC=+0.872, citations [1])
        are intentionally preserved.
        """
        if not text:
            return ""
        text = re.sub(r"\bPage\s+\d+(\s+of\s+\d+)?\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def remove_special_characters(self, text: str) -> str:
        """
        Remove only genuinely harmful characters:
        - Null bytes
        - ASCII control characters
        - Unicode replacement character U+FFFD

        Intentionally KEEPS: + = [ ] % / # @ _ * & ^ ~ < > | quotes
        These appear in correlation values (CC=+0.872), citations ([1]),
        technical terms (Grad-CAM, fMRI), and mathematical notation.
        """
        if not text:
            return ""
        text = text.replace("\x00", "").replace("\ufffd", "")
        text = re.sub(r"[\x01-\x08\x0b-\x1f\x7f]", "", text)
        return text.strip()

    def normalize_text(self, text: str) -> str:
        """
        NFKD Unicode normalisation — resolves ligatures and composed forms.
        Does NOT lowercase — case preserves acronyms (XAI, fMRI, CNN, ViT).
        """
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text)
        return text.strip()

    def clean_text(self, text: Optional[str]) -> str:
        """Full cleaning pipeline: whitespace → headers/footers → control chars → NFKD."""
        if not text:
            return ""
        text = self.remove_extra_whitespace(text)
        text = self.remove_headers_footers(text)
        text = self.remove_special_characters(text)
        text = self.normalize_text(text)
        return text