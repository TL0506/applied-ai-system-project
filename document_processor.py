import logging
import os

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class DocumentProcessingError(Exception):
    pass


class FileTooLargeError(DocumentProcessingError):
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    pass


class EmptyDocumentError(DocumentProcessingError):
    pass


class DocumentProcessor:

    def validate_file(self, file_name: str, file_size_bytes: int) -> None:
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File type '{ext}' is not supported. Please upload a PDF or TXT file."
            )
        if file_size_bytes > MAX_FILE_SIZE_BYTES:
            size_mb = file_size_bytes / (1024 * 1024)
            raise FileTooLargeError(
                f"File is {size_mb:.1f} MB. Maximum allowed size is 10 MB."
            )
        logger.info("File validated: %s (%.1f KB)", file_name, file_size_bytes / 1024)

    def extract_text(self, uploaded_file) -> str:
        name = uploaded_file.name
        # Streamlit UploadedFile exposes .size; fall back to reading bytes
        size = getattr(uploaded_file, "size", None)
        if size is None:
            data = uploaded_file.read()
            uploaded_file.seek(0)
            size = len(data)

        self.validate_file(name, size)

        ext = os.path.splitext(name)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(uploaded_file)
        return self._extract_txt(uploaded_file)

    def _extract_pdf(self, uploaded_file) -> str:
        try:
            import PyPDF2
        except ImportError:
            raise DocumentProcessingError(
                "PyPDF2 is not installed. Run: pip install PyPDF2"
            )

        try:
            reader = PyPDF2.PdfReader(uploaded_file)
        except Exception as e:
            raise DocumentProcessingError(f"Could not open PDF: {e}") from e

        if reader.is_encrypted:
            raise DocumentProcessingError(
                "This PDF is encrypted. Please provide an unencrypted PDF."
            )

        pages_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                pages_text.append(text)
            except Exception as e:
                logger.warning("Could not extract text from page %d: %s", i + 1, e)

        full_text = "\n".join(pages_text).strip()
        if not full_text:
            raise EmptyDocumentError(
                "No text could be extracted from this PDF. "
                "It may be a scanned image PDF, which requires OCR (not supported)."
            )

        logger.info(
            "Extracted %d chars from PDF (%d pages)", len(full_text), len(reader.pages)
        )
        return full_text

    def _extract_txt(self, uploaded_file) -> str:
        raw_bytes = uploaded_file.read()
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        if not text:
            raise EmptyDocumentError("The uploaded text file is empty.")
        logger.info("Extracted %d chars from TXT file", len(text))
        return text
