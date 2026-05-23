# backend/pdf_processor.py
"""
PDF text extraction with automatic fallback.
Primary extractor : PyMuPDF  (fast, accurate)
Fallback extractor: pdfplumber (handles edge-case PDFs)
"""

import logging
from typing import List, Dict, Any, BinaryIO

# PyMuPDF: optional — DLL load failures on Windows are caught silently
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except (ImportError, OSError):
    PYMUPDF_AVAILABLE = False
import pdfplumber

logger = logging.getLogger("smartdocs.pdf_processor")


class PDFProcessingError(Exception):
    """Raised when PDF extraction or parsing fails."""
    pass


class PDFProcessor:
    """
    Handles PDF text extraction and metadata retrieval.
    Works with Streamlit UploadedFile objects or any binary file-like object.
    """

    def extract_text_pymupdf(self, file: BinaryIO) -> List[Dict[str, Any]]:
        if not PYMUPDF_AVAILABLE:
            raise PDFProcessingError("PyMuPDF not available on this system.")
        pages: List[Dict[str, Any]] = []
        try:
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if doc.is_encrypted:
                raise PDFProcessingError("PDF is password-protected.")
            if doc.page_count == 0:
                raise PDFProcessingError("PDF has no pages.")
            for idx in range(doc.page_count):
                text = doc.load_page(idx).get_text().strip()
                pages.append({"page_number": idx + 1, "text": text})
            doc.close()
            if all(not p["text"] for p in pages):
                raise PDFProcessingError("PDF contains no extractable text.")
            return pages
        except fitz.FileDataError:
            raise PDFProcessingError("Invalid or corrupted PDF file.")
        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"PyMuPDF extraction failed: {e}")

    def extract_text_pdfplumber(self, file: BinaryIO) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(file) as pdf:
                if not pdf.pages:
                    raise PDFProcessingError("PDF has no pages.")
                for i, page in enumerate(pdf.pages):
                    text = (page.extract_text() or "").strip()
                    pages.append({"page_number": i + 1, "text": text})
            if all(not p["text"] for p in pages):
                raise PDFProcessingError("PDF contains no extractable text.")
            return pages
        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"pdfplumber extraction failed: {e}")

    def get_pdf_metadata(self, file: BinaryIO) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            raise PDFProcessingError("PyMuPDF not available on this system.")
        try:
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            meta = doc.metadata or {}
            result = {
                "page_count": doc.page_count,
                "title":  meta.get("title")  or "Unknown",
                "author": meta.get("author") or "Unknown",
            }
            doc.close()
            return result
        except fitz.FileDataError:
            raise PDFProcessingError("Invalid or corrupted PDF file.")
        except Exception as e:
            raise PDFProcessingError(f"Metadata extraction failed: {e}")

    def extract_text(self, uploaded_file) -> List[Dict[str, Any]]:
        """Main entry point. Tries PyMuPDF if available, falls back to pdfplumber."""
        uploaded_file.seek(0)
        if PYMUPDF_AVAILABLE:
            try:
                pages = self.extract_text_pymupdf(uploaded_file)
                logger.debug(f"PyMuPDF extracted {len(pages)} pages")
                return pages
            except PDFProcessingError as primary_err:
                logger.warning(f"PyMuPDF failed ({primary_err}), retrying with pdfplumber")
                uploaded_file.seek(0)
        try:
            pages = self.extract_text_pdfplumber(uploaded_file)
            logger.debug(f"pdfplumber extracted {len(pages)} pages")
            return pages
        except PDFProcessingError as fallback_err:
            raise PDFProcessingError(f"pdfplumber extraction failed: {fallback_err}.")