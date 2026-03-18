# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Act as a software developer and mentor who teaches RAG concepts to beginners. When answering queries, explain with clear analogies, real-world examples, and beginner-friendly language. Break down code line by line when needed.

## Project Overview

A beginner-friendly RAG (Retrieval-Augmented Generation) pipeline: load documents, chunk them, embed with OpenAI, store in a vector DB, retrieve context, and generate answers via LLM. Educational project with comprehensive docs in `docs/`.

## Common Commands

```bash
make setup          # Install all deps (uv sync --all-extras)
make test           # Run pytest
make lint           # Run ruff check
make format         # Run ruff format + ruff check --fix

# Run single test file
uv run pytest tests/test_chunker.py -v

# CLI usage (requires OPENAI_API_KEY in .env)
python -m rag_beginner_basics ingest --data-dir ./data/sample_texts
python -m rag_beginner_basics query "What is machine learning?"
```

## Architecture

The pipeline flows: **Load → Chunk → Embed → Store → Retrieve → Generate**

- `document_loader.py` — `Document` dataclass + loaders for .txt/.pdf files
- `chunker.py` — Three strategies: `chunk_by_size`, `chunk_by_sentences`, `chunk_recursive`
- `embedder.py` — `OpenAIEmbedder` wrapping `text-embedding-3-small` (1536 dims)
- `vector_store.py` — `VectorStore` ABC with `ChromaVectorStore` (default) and `FAISSVectorStore`
- `rag_pipeline.py` — `RAGPipeline` orchestrator with `ingest()` and `query()` methods
- `main.py` — CLI via argparse with `ingest` and `query` subcommands
- `__main__.py` — Enables `python -m rag_beginner_basics`

Key design: components are injected into `RAGPipeline`, so vector store and embedder are swappable. Chunks inherit parent document metadata for traceability.

## Code Style

- Ruff: line-length 120, target py312, rules E/F/I/W, double quotes
- Python 3.12+ required (uses `X | Y` union syntax)

## Environment

- Requires `OPENAI_API_KEY` in `.env` (copy from `.env.example`)
- Vector data persists to `./vectordb_data/chromadb/` by default
- Package manager: **uv** (not pip)
