"""Tests for document loading utilities."""

import tempfile
from pathlib import Path

from rag_beginner_basics.document_loader import Document, load_directory, load_text_file


def test_document_creation():
    doc = Document(content="Hello world", metadata={"source": "test.txt"})
    assert doc.content == "Hello world"
    assert doc.metadata["source"] == "test.txt"


def test_document_default_metadata():
    doc = Document(content="Hello")
    assert doc.metadata == {}


def test_load_text_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is test content.")
        f.flush()

        doc = load_text_file(f.name)
        assert doc.content == "This is test content."
        assert doc.metadata["source"] == Path(f.name).name
        assert doc.metadata["file_type"] == ".txt"
        assert doc.metadata["char_count"] == 21

    Path(f.name).unlink()


def test_load_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        for name in ["a.txt", "b.txt", "c.md"]:
            Path(tmpdir, name).write_text(f"Content of {name}")

        # Load only .txt files
        docs = load_directory(tmpdir, pattern="*.txt")
        assert len(docs) == 2
        sources = [d.metadata["source"] for d in docs]
        assert "a.txt" in sources
        assert "b.txt" in sources


def test_load_directory_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        docs = load_directory(tmpdir, pattern="*.txt")
        assert len(docs) == 0
