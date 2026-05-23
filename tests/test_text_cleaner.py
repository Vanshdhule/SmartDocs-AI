# tests/test_text_cleaner.py
"""Unit tests for TextCleaner — no external dependencies required."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from backend.text_cleaner import TextCleaner

@pytest.fixture
def cleaner():
    return TextCleaner()

class TestRemoveExtraWhitespace:
    def test_collapses_multiple_spaces(self, cleaner):
        assert cleaner.remove_extra_whitespace("hello   world") == "hello world"
    def test_collapses_newlines(self, cleaner):
        assert cleaner.remove_extra_whitespace("hello\n\nworld") == "hello world"
    def test_empty_string(self, cleaner):
        assert cleaner.remove_extra_whitespace("") == ""
    def test_only_whitespace(self, cleaner):
        assert cleaner.remove_extra_whitespace("   \n\t  ") == ""

class TestRemoveHeadersFooters:
    def test_removes_page_n(self, cleaner):
        result = cleaner.remove_headers_footers("Introduction\nPage 5\nContent here")
        assert "Page 5" not in result
        assert "Content here" in result
    def test_removes_page_n_of_m(self, cleaner):
        result = cleaner.remove_headers_footers("Page 3 of 10\nSome text")
        assert "Page 3 of 10" not in result
    def test_preserves_years_in_sentences(self, cleaner):
        text = "In 2023, revenue grew by 42%."
        assert "2023" in cleaner.remove_headers_footers(text)
    def test_preserves_correlation_values(self, cleaner):
        text = "The result CC=+0.872 is significant."
        assert "0.872" in cleaner.remove_headers_footers(text)
    def test_removes_bare_page_number_line(self, cleaner):
        text = "Conclusion\n   12   \nThis chapter summarises"
        result = cleaner.remove_headers_footers(text)
        assert "Conclusion" in result
        assert "This chapter summarises" in result
    def test_empty_string(self, cleaner):
        assert cleaner.remove_headers_footers("") == ""

class TestRemoveSpecialCharacters:
    def test_removes_null_bytes(self, cleaner):
        result = cleaner.remove_special_characters("hello\x00world")
        assert "\x00" not in result
        assert "hello" in result
    def test_removes_replacement_char(self, cleaner):
        assert "\ufffd" not in cleaner.remove_special_characters("hello\ufffdworld")
    def test_keeps_brackets_and_equals(self, cleaner):
        text = "CC=+0.872 and [Source: file.pdf, Page: 4]"
        result = cleaner.remove_special_characters(text)
        assert "=" in result
        assert "[" in result
        assert "]" in result
    def test_keeps_technical_terms(self, cleaner):
        text = "Grad-CAM, Score-CAM, fMRI [1]"
        result = cleaner.remove_special_characters(text)
        assert "Grad-CAM" in result
        assert "[1]" in result
    def test_empty_string(self, cleaner):
        assert cleaner.remove_special_characters("") == ""

class TestNormalizeText:
    def test_does_not_lowercase(self, cleaner):
        result = cleaner.normalize_text("NASA and WHO are acronyms.")
        assert "NASA" in result
        assert "WHO" in result
    def test_empty_string(self, cleaner):
        assert cleaner.normalize_text("") == ""

class TestCleanText:
    def test_removes_page_markers(self, cleaner):
        dirty = "  Hello   World \n\nPage 3 of 10\n  "
        assert "Page 3 of 10" not in cleaner.clean_text(dirty)
    def test_preserves_technical_content(self, cleaner):
        text = "Result CC=+0.872 cited in [Source: report.pdf, Page: 12]."
        result = cleaner.clean_text(text)
        assert "0.872" in result
        assert "report.pdf" in result
    def test_none_input(self, cleaner):
        assert cleaner.clean_text(None) == ""
    def test_empty_input(self, cleaner):
        assert cleaner.clean_text("") == ""
    def test_preserves_acronyms(self, cleaner):
        result = cleaner.clean_text("XAI, fMRI, CNN, and ViT are ML acronyms.")
        assert "XAI" in result
        assert "fMRI" in result