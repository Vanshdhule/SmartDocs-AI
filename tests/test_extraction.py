# test_extraction.py

from backend.pdf_processor import PDFProcessor, PDFProcessingError

PDF_FILES = [
    r"uploads\Albert_Einstein,_John_Stachel,_Roger_Penrose_Einstein's_Miraculous.pdf"
]


def test_pdf_extraction():
    processor = PDFProcessor()
    
    for pdf_path in PDF_FILES:
        print("=" * 80)
        print(f"Testing PDF: {pdf_path}")

        try:
            # Open the PDF file and pass the file object
            with open(pdf_path, "rb") as pdf_file:
                pages = processor.extract_text(pdf_file)

            print(f"Total Pages  : {len(pages)}")
            print("-" * 40)

            for page_data in pages:
                page_num = page_data["page_number"]
                text = page_data["text"]
                preview = text[:300].replace("\n", " ")
                print(f"[Page {page_num}] {preview}...")

        except PDFProcessingError as e:
            print(f"ERROR: {e}")

        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_pdf_extraction()