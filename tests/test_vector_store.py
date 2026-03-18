"""Tests for vector store implementations (no API calls required)."""

import tempfile

import numpy as np

from rag_beginner_basics.document_loader import Document
from rag_beginner_basics.vector_store import FAISSVectorStore


def _random_embeddings(n: int, dim: int = 1536) -> list[list[float]]:
    """Generate random embeddings for testing."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    return vecs.tolist()


def test_faiss_add_and_count():
    store = FAISSVectorStore(dimension=1536)
    docs = [Document(content=f"Doc {i}", metadata={"id": i}) for i in range(5)]
    embeddings = _random_embeddings(5)

    store.add(docs, embeddings)
    assert store.count() == 5


def test_faiss_query():
    store = FAISSVectorStore(dimension=1536)
    docs = [
        Document(content="Python is a programming language", metadata={"topic": "python"}),
        Document(content="Machine learning uses data", metadata={"topic": "ml"}),
        Document(content="Cloud computing provides services", metadata={"topic": "cloud"}),
    ]
    embeddings = _random_embeddings(3)
    store.add(docs, embeddings)

    results = store.query(embeddings[0], top_k=2)
    assert len(results) == 2
    assert "content" in results[0]
    assert "metadata" in results[0]
    assert "similarity" in results[0]


def test_faiss_query_with_filter():
    store = FAISSVectorStore(dimension=1536)
    docs = [
        Document(content="Doc A", metadata={"source": "a.txt"}),
        Document(content="Doc B", metadata={"source": "b.txt"}),
        Document(content="Doc C", metadata={"source": "a.txt"}),
    ]
    embeddings = _random_embeddings(3)
    store.add(docs, embeddings)

    results = store.query(
        embeddings[0],
        top_k=3,
        filter_fn=lambda m: m.get("source") == "a.txt",
    )
    assert all(r["metadata"]["source"] == "a.txt" for r in results)


def test_faiss_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and populate store
        store1 = FAISSVectorStore(dimension=1536, persist_dir=tmpdir)
        docs = [Document(content=f"Doc {i}", metadata={"id": i}) for i in range(3)]
        embeddings = _random_embeddings(3)
        store1.add(docs, embeddings)
        assert store1.count() == 3

        # Load from disk
        store2 = FAISSVectorStore(dimension=1536, persist_dir=tmpdir)
        assert store2.count() == 3

        results = store2.query(embeddings[0], top_k=1)
        assert len(results) == 1


def test_faiss_empty_store():
    store = FAISSVectorStore(dimension=1536)
    assert store.count() == 0
