"""Vector store implementations for ChromaDB and FAISS."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Callable

import chromadb
import faiss
import numpy as np

from rag_beginner_basics.document_loader import Document


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with embeddings to the store."""

    @abstractmethod
    def query(self, query_embedding: list[float], top_k: int = 3) -> list[dict]:
        """Query the store for similar documents."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the store."""


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store with built-in persistence and metadata support."""

    def __init__(self, collection_name: str = "default", persist_dir: str = "./vectordb_data/chromadb"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with embeddings to ChromaDB."""
        ids = []
        texts = []
        metadatas = []

        for doc in documents:
            doc_id = hashlib.md5(doc.content.encode("utf-8")).hexdigest()
            ids.append(doc_id)
            texts.append(doc.content)
            metadatas.append(doc.metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, query_embedding: list[float], top_k: int = 3, where: dict | None = None) -> list[dict]:
        """Query ChromaDB for similar documents."""
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection.name)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store with manual metadata management and persistence."""

    def __init__(self, dimension: int = 1536, persist_dir: str | None = None):
        self.dimension = dimension
        self.persist_dir = persist_dir
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_store: list[dict] = []

        if persist_dir:
            self._load()

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Add documents with embeddings to FAISS."""
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        for doc in documents:
            self.doc_store.append({"content": doc.content, "metadata": doc.metadata})

        if self.persist_dir:
            self._save()

    def query(
        self, query_embedding: list[float], top_k: int = 3, filter_fn: Callable | None = None
    ) -> list[dict]:
        """Search FAISS for similar documents."""
        query_vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        search_k = top_k * 3 if filter_fn else top_k
        scores, indices = self.index.search(query_vec, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = self.doc_store[idx]
            if filter_fn and not filter_fn(entry["metadata"]):
                continue
            results.append({
                "content": entry["content"],
                "metadata": entry["metadata"],
                "similarity": float(score),
            })
            if len(results) >= top_k:
                break
        return results

    def count(self) -> int:
        return self.index.ntotal

    def _save(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.persist_dir, "index.faiss"))
        with open(os.path.join(self.persist_dir, "doc_store.json"), "w") as f:
            json.dump(self.doc_store, f)

    def _load(self) -> None:
        index_path = os.path.join(self.persist_dir, "index.faiss")
        store_path = os.path.join(self.persist_dir, "doc_store.json")
        if os.path.exists(index_path) and os.path.exists(store_path):
            self.index = faiss.read_index(index_path)
            with open(store_path) as f:
                self.doc_store = json.load(f)
