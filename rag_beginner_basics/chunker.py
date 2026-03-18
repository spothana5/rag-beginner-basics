"""Text chunking strategies for splitting documents into smaller pieces."""

import re

from rag_beginner_basics.document_loader import Document


def chunk_by_size(doc: Document, chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    """Split document into fixed-size chunks with overlap."""
    text = doc.content
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunk_metadata = {
            **doc.metadata,
            "chunk_index": chunk_index,
            "chunk_method": "fixed_size",
            "chunk_char_count": len(chunk_text),
        }
        chunks.append(Document(content=chunk_text, metadata=chunk_metadata))

        start += chunk_size - overlap
        chunk_index += 1

    return chunks


def chunk_by_sentences(doc: Document, sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> list[Document]:
    """Split document into chunks based on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", doc.content)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(sentences):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk_text = " ".join(sentences[start:end])

        chunk_metadata = {
            **doc.metadata,
            "chunk_index": chunk_index,
            "chunk_method": "sentence",
            "num_sentences": end - start,
            "chunk_char_count": len(chunk_text),
        }
        chunks.append(Document(content=chunk_text, metadata=chunk_metadata))

        start += sentences_per_chunk - overlap_sentences
        chunk_index += 1

    return chunks


def chunk_recursive(
    doc: Document,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] | None = None,
) -> list[Document]:
    """Recursively split text using a hierarchy of separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split_text(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        sep = seps[0]
        parts = text.split(sep)

        result = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if len(part) > chunk_size:
                    result.extend(_split_text(part, seps[1:]))
                    current = ""
                else:
                    current = part

        if current:
            result.append(current)

        return result

    pieces = _split_text(doc.content, separators)
    chunks = []
    for i, piece in enumerate(pieces):
        chunk_metadata = {
            **doc.metadata,
            "chunk_index": i,
            "chunk_method": "recursive",
            "chunk_char_count": len(piece),
        }
        chunks.append(Document(content=piece.strip(), metadata=chunk_metadata))

    return chunks
