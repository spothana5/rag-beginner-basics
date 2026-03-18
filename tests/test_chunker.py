"""Tests for text chunking strategies."""

from rag_beginner_basics.chunker import chunk_by_sentences, chunk_by_size, chunk_recursive
from rag_beginner_basics.document_loader import Document


def _make_doc(length: int = 1000) -> Document:
    """Create a test document of approximately the given length."""
    text = "This is a sentence. " * (length // 20)
    return Document(content=text.strip(), metadata={"source": "test.txt"})


def test_chunk_by_size_basic():
    doc = _make_doc(1000)
    chunks = chunk_by_size(doc, chunk_size=200, overlap=0)
    assert len(chunks) > 1
    assert all(len(c.content) <= 200 for c in chunks)


def test_chunk_by_size_with_overlap():
    doc = Document(content="A" * 500, metadata={"source": "test.txt"})
    chunks = chunk_by_size(doc, chunk_size=200, overlap=50)
    # With overlap, we get more chunks
    no_overlap_chunks = chunk_by_size(doc, chunk_size=200, overlap=0)
    assert len(chunks) >= len(no_overlap_chunks)


def test_chunk_by_size_preserves_metadata():
    doc = Document(content="Hello world " * 100, metadata={"source": "test.txt", "topic": "greeting"})
    chunks = chunk_by_size(doc, chunk_size=100)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
        assert chunk.metadata["topic"] == "greeting"
        assert "chunk_index" in chunk.metadata


def test_chunk_by_size_small_doc():
    doc = Document(content="Short text.", metadata={})
    chunks = chunk_by_size(doc, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].content == "Short text."


def test_chunk_by_sentences_basic():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    doc = Document(content=text, metadata={})
    chunks = chunk_by_sentences(doc, sentences_per_chunk=2, overlap_sentences=0)
    assert len(chunks) >= 2


def test_chunk_by_sentences_with_overlap():
    text = "One. Two. Three. Four. Five. Six. Seven. Eight."
    doc = Document(content=text, metadata={})
    chunks = chunk_by_sentences(doc, sentences_per_chunk=3, overlap_sentences=1)
    assert len(chunks) >= 2
    assert all("num_sentences" in c.metadata for c in chunks)


def test_chunk_recursive_basic():
    doc = _make_doc(1000)
    chunks = chunk_recursive(doc, chunk_size=200)
    assert len(chunks) > 1
    assert all(c.metadata["chunk_method"] == "recursive" for c in chunks)


def test_chunk_recursive_respects_size():
    doc = _make_doc(2000)
    chunks = chunk_recursive(doc, chunk_size=300)
    # Most chunks should be near or under the size limit
    oversized = [c for c in chunks if len(c.content) > 300 * 1.5]
    assert len(oversized) == 0, f"Found {len(oversized)} oversized chunks"
