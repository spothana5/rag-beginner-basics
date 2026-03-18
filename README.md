# RAG Beginner Basics

A beginner-friendly, hands-on project that implements a complete **Retrieval-Augmented Generation (RAG)** pipeline from scratch using Python.

> **What is RAG?** Think of an LLM as a brilliant student taking an open-book exam instead of relying on memory alone. RAG gives the LLM the right pages to look at before answering — so it stays grounded in real data instead of hallucinating.

---

## What You'll Learn

- **Document Loading** — Read `.txt` and `.pdf` files into structured Python objects
- **Text Chunking** — Split documents using 3 strategies (fixed-size, sentence-based, recursive)
- **Embeddings** — Convert text into 1536-dimensional vectors using OpenAI's `text-embedding-3-small`
- **Vector Stores** — Store and search embeddings with ChromaDB and FAISS
- **RAG Pipeline** — Retrieve relevant context and generate answers using an LLM
- **CLI Interface** — Ingest documents and query from the terminal

---

## RAG Pipeline Architecture

```
                        INGEST (one-time setup)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  Load    │ -> │  Chunk   │ -> │  Embed   │ -> │  Store in    │
│  Files   │    │  Text    │    │  Vectors │    │  Vector DB   │
└──────────┘    └──────────┘    └──────────┘    └──────────────┘

                         QUERY (every question)
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
│  User    │ -> │  Embed   │ -> │ Retrieve │ -> │  LLM Answers │
│ Question │    │  Query   │    │ Context  │    │  from Context│
└──────────┘    └──────────┘    └──────────┘    └──────────────┘
```

---

## Prerequisites

- **Python 3.12+**
- **uv** (Python package manager) — [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
- **OpenAI API key** — [Get one here](https://platform.openai.com/api-keys)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-beginner-basics.git
cd rag-beginner-basics
```

### 2. Install dependencies

```bash
uv sync --all-extras
```

### 3. Activate the virtual environment

```bash
source .venv/bin/activate
```

### 4. Set up your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 5. Ingest sample documents

```bash
python -m rag_beginner_basics ingest --data-dir ./data/sample_texts
```

### 6. Ask a question

```bash
python -m rag_beginner_basics query "What is machine learning?"
```

---

## Project Structure

```
rag-beginner-basics/
├── rag_beginner_basics/        # Main Python package
│   ├── __init__.py             # Package version
│   ├── __main__.py             # python -m entry point
│   ├── document_loader.py      # Load .txt and .pdf files
│   ├── chunker.py              # 3 chunking strategies
│   ├── embedder.py             # OpenAI embeddings wrapper
│   ├── vector_store.py         # ChromaDB + FAISS implementations
│   ├── rag_pipeline.py         # End-to-end RAG orchestrator
│   └── main.py                 # CLI with ingest/query commands
├── tests/                      # Unit tests
├── data/sample_texts/          # Sample documents for testing
├── docs/                       # Detailed explanations of each module
├── .env.example                # Environment variables template
├── .gitignore                  # Git ignore rules
├── CLAUDE.md                   # Claude Code guidance
├── pyproject.toml              # Dependencies and tool config
└── Makefile                    # Development shortcuts
```

---

## CLI Usage

### Ingest documents

```bash
python -m rag_beginner_basics ingest --data-dir ./data/sample_texts
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | **(required)** | Path to folder containing documents |
| `--pattern` | `*.txt` | File pattern to match |
| `--chunk-size` | `500` | Characters per chunk |
| `--overlap` | `50` | Overlap between chunks |
| `--collection` | `rag-basics` | ChromaDB collection name |
| `--persist-dir` | `./vectordb_data/chromadb` | ChromaDB persist directory |

### Query documents

```bash
python -m rag_beginner_basics query "What are the benefits of cloud computing?"
```

| Flag | Default | Description |
|------|---------|-------------|
| `question` | **(required)** | The question to ask |
| `--top-k` | `3` | Number of chunks to retrieve |
| `--model` | `gpt-4o-mini` | OpenAI model for generation |
| `--collection` | `rag-basics` | ChromaDB collection name |
| `--persist-dir` | `./vectordb_data/chromadb` | ChromaDB persist directory |

---

## Development

```bash
make setup       # Install all dependencies
make test        # Run tests
make lint        # Check code style
make format      # Auto-format code
make clean       # Remove generated files
```

---

## Documentation

Each module has a detailed explanation in `docs/` with analogies, line-by-line breakdowns, and real-world examples:

| # | Doc | Covers |
|---|-----|--------|
| 01 | [Document Loader](docs/01.DOCUMENT_LOADER_EXPLANATION.md) | Loading files, `Document` dataclass, metadata |
| 02 | [Chunker](docs/02.CHUNKER_EXPLANATION.md) | Fixed-size, sentence, and recursive chunking |
| 03 | [Embedder](docs/03.EMBEDDER_EXPLANATION.md) | OpenAI embeddings, vector dimensions, batching |
| 04 | [Vector Store](docs/04.VECTOR_STORE_EXPLANATION.md) | ChromaDB vs FAISS, similarity search, persistence |
| 05 | [RAG Pipeline](docs/05.RAG_PIPELINE_EXPLANATION.md) | Orchestration, prompt template, LLM generation |
| 06 | [Main CLI](docs/06.MAIN_EXPLANATION.md) | argparse, subcommands, entry points |
| 07 | [End-to-End RAG Flow](docs/07.END_TO_END_RAG_FLOW.md) | Complete pipeline trace with diagrams |
| 08 | [Testing the CLI](docs/08.TESTING_THE_CLI.md) | Running ingest/query, troubleshooting |

For a complete step-by-step guide to build this project from scratch, see [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12+ |
| Package Manager | uv |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector Store | ChromaDB (default), FAISS |
| PDF Parsing | pypdf |
| Testing | pytest |
| Linting | ruff |

---

## Sample Output

```
$ python -m rag_beginner_basics query "What is machine learning?"

Question: What is machine learning?

Answer: Machine learning is a subset of artificial intelligence that enables
computers to learn from data and improve their performance on tasks without
being explicitly programmed.

Sources: machine_learning.txt, machine_learning.txt, machine_learning.txt
```

---

## License

This is an educational project. Feel free to use, modify, and learn from it.
