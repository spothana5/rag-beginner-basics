"""Document loading utilities for text and PDF files."""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Document:
    """A document with content and metadata."""

    content: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


def load_text_file(file_path: str) -> Document:
    """Load a text file into a Document."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    metadata = {
        "source": path.name,
        "file_path": str(path),
        "file_type": path.suffix,
        "char_count": len(content),
    }
    return Document(content=content, metadata=metadata)


def load_pdf_file(file_path: str) -> Document:
    """Load a PDF file into a Document."""
    path = Path(file_path)
    reader = PdfReader(str(path))

    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    content = "\n\n".join(pages_text)
    metadata = {
        "source": path.name,
        "file_path": str(path),
        "file_type": ".pdf",
        "num_pages": len(reader.pages),
        "char_count": len(content),
    }
    return Document(content=content, metadata=metadata)


def load_directory(dir_path: str, pattern: str = "*.txt") -> list[Document]:
    """Load all matching files from a directory."""
    path = Path(dir_path)
    documents = []
    for file_path in sorted(path.glob(pattern)):
        if file_path.suffix == ".pdf":
            doc = load_pdf_file(str(file_path))
        else:
            doc = load_text_file(str(file_path))
        documents.append(doc)
    return documents
