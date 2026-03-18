# RAG Beginner Basics — Hands-On Training Guide

This guide walks you through building a RAG (Retrieval-Augmented Generation) project **from scratch**. Every command is meant to be run by YOU in your terminal, step by step.

## Table of contents — Step reference links

| # | Step |
|---|------|
| 1 | [Step 1: Check Python version](#step-1-check-python-version) |
| 2 | [Step 2: Install uv (Python package manager)](#step-2-install-uv-python-package-manager) |
| 3 | [Step 3: Create the project directory](#step-3-create-the-project-directory) |
| 7 | [Step 7: Create the project structure](#step-7-create-the-project-structure) |
| 8 | [Step 8: Verify the structure](#step-8-verify-the-structure) |
| 9 | [Step 9: Create `pyproject.toml`](#step-9-create-pyprojecttoml) |
| 10 | [Step 10: Create `README.md`](#step-10-create-readmemd) |
| 11 | [Step 11: Create `.env.example`](#step-11-create-envexample) |
| 12 | [Step 12: Create `.gitignore`](#step-12-create-gitignore) |
| 13 | [Step 13: Create `Makefile`](#step-13-create-makefile) |
| 14 | [Step 14: Verify your config files exist](#step-14-verify-your-config-files-exist) |
| 15 | [Step 15: Create `rag_beginner_basics/__init__.py` (required before install)](#step-15-create-rag_beginner_basics__init__py-required-before-install) |
| 16 | [Step 16: Install all dependencies using uv](#step-16-install-all-dependencies-using-uv) |
| 16b | [Step 16b: Activate the virtual environment](#step-16b-activate-the-virtual-environment) |
| 17 | [Step 17: Verify the installation](#step-17-verify-the-installation) |
| 18 | [Step 18: Create sample text files](#step-18-create-sample-text-files) |
| 19 | [Step 19: Verify your data files](#step-19-verify-your-data-files) |
| 20 | [Step 20: Add content to `rag_beginner_basics/__init__.py`](#step-20-add-content-to-rag_beginner_basics__init__py) |
| 21 | [Step 21: Create `rag_beginner_basics/document_loader.py`](#step-21-create-rag_beginner_basicsdocument_loaderpy) |
| 22 | [Step 22: Test your document loader manually](#step-22-test-your-document-loader-manually) |
| 23 | [Step 23: Create `rag_beginner_basics/chunker.py`](#step-23-create-rag_beginner_basicschunkerpy) |
| 24 | [Step 24: Test your chunker manually](#step-24-test-your-chunker-manually) |
| 25 | [Step 25: Create `rag_beginner_basics/embedder.py`](#step-25-create-rag_beginner_basicsembedderpy) |
| 26 | [Step 26: Create `rag_beginner_basics/vector_store.py`](#step-26-create-rag_beginner_basicsvector_storepy) |
| 27 | [Step 27: Create `rag_beginner_basics/rag_pipeline.py`](#step-27-create-rag_beginner_basicsrag_pipelinepy) |
| 28 | [Step 28: Create `rag_beginner_basics/main.py`](#step-28-create-rag_beginner_basicsmainpy) |
| 28b | [Step 28b: Create `rag_beginner_basics/__main__.py`](#step-28b-create-rag_beginner_basics__main__py) |
| 29 | [Step 29: Verify the package structure](#step-29-verify-the-package-structure) |
| 30 | [Step 30: Create `tests/__init__.py`](#step-30-create-tests__init__py) |
| 31 | [Step 31: Create `tests/test_document_loader.py`](#step-31-create-teststest_document_loaderpy) |
| 32 | [Step 32: Create `tests/test_chunker.py`](#step-32-create-teststest_chunkerpy) |
| 33 | [Step 33: Create `tests/test_vector_store.py`](#step-33-create-teststest_vector_storepy) |
| 34 | [Step 34: Run unit tests](#step-34-run-unit-tests) |
| 35 | [Step 35: Run linter](#step-35-run-linter) |
| 36 | [Step 36: Format code](#step-36-format-code) |
| 37 | [Step 37: Run tests again to make sure formatting didn't break anything](#step-37-run-tests-again-to-make-sure-formatting-didnt-break-anything) |
| 38 | [Step 38: Use the Makefile shortcuts](#step-38-use-the-makefile-shortcuts) |
| 39 | [Step 39: Set up your API key](#step-39-set-up-your-api-key) |
| 40 | [Step 40: Test the CLI ingest command](#step-40-test-the-cli-ingest-command) |
| 41 | [Step 41: Test the CLI query command](#step-41-test-the-cli-query-command) |
| 42 | [Step 42: Try different queries](#step-42-try-different-queries) |

---

## Prerequisites

Before starting, make sure you have these installed on your machine.

### Step 1: Check Python version

```bash
python3 --version
```

You need **Python 3.12+**. If not installed:

```bash
# macOS (using Homebrew)
brew install python@3.12
```

### Step 2: Install uv (Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify it installed:

```bash
uv --version
```

---

## Phase 1: Project Setup

### Step 3: Create the project directory

```bash
mkdir -p /Users/spothana/personal_projects/rag-beginner-basics
cd /Users/spothana/personal_projects/rag-beginner-basics
```

### Step 4: Create the project structure

```bash
mkdir -p rag_beginner_basics
mkdir -p tests
mkdir -p data/sample_texts
mkdir -p docs
```

### Step 8: Verify the structure

```bash
ls -la
ls -la rag_beginner_basics/
ls -la tests/
ls -la data/sample_texts/
```

---

## Phase 2: Project Configuration Files

### Step 9: Create `pyproject.toml`

This is the heart of your Python project. Open your editor and create the file:

```bash
# Using VS Code:
code pyproject.toml

# Or using nano:
nano pyproject.toml
```

Paste this content:

```toml
[project]
name = "rag-beginner-basics"
version = "0.1.0"
description = "A beginner-friendly RAG learning repository"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.50.0",
    "chromadb>=1.3.0",
    "faiss-cpu>=1.9.0",
    "numpy>=2.0.0",
    "pypdf>=4.0.0",
    "python-dotenv>=1.0.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "jupyter>=1.1.0",
    "matplotlib>=3.9.0",
    "scikit-learn>=1.5.0",
    "ipywidgets>=8.0.0",
    "ruff>=0.3.0",
    "pytest>=8.0.0",
    "black>=24.0.0",
]

[project.scripts]
rag-beginner-basics = "rag_beginner_basics.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 120
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.ruff.format]
quote-style = "double"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Save and close the file.

### Step 10: Create `README.md`

Your `pyproject.toml` references `readme = "README.md"`, so this file **must exist** before you install dependencies. Without it, `uv sync` will fail with `OSError: Readme file does not exist: README.md`.

```bash
echo "# RAG Beginner Basics" > README.md
```

You'll add the full content later in Phase 12 (Step 55). For now, this one-liner is enough.

### Step 11: Create `.env.example`

```bash
code .env.example
```

Paste:

```
OPENAI_API_KEY=sk-your-api-key-here
```

### Step 12: Create `.gitignore`

```bash
code .gitignore
```

Paste:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# Virtual environment
.venv/

# Environment variables
.env

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# Vector DB data (generated at runtime)
vectordb_data/

# OS
.DS_Store
Thumbs.db

# uv
uv.lock
```

### Step 13: Create `Makefile`

```bash
code Makefile
```

Paste (IMPORTANT — Makefile requires **tabs**, not spaces, before commands):

```makefile
.PHONY: setup dev-install test lint format jupyter docker-build docker-run docker-shell clean

# ── Standalone Setup ──────────────────────────────────────────────
setup:
	uv sync --all-extras

dev-install:
	uv sync --all-extras

# ── Development ───────────────────────────────────────────────────
test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

jupyter:
	uv run jupyter notebook --notebook-dir=notebooks

# ── Docker ────────────────────────────────────────────────────────
docker-build:
	docker build -t rag-beginner-basics .

docker-run:
	docker compose up

docker-shell:
	docker compose run --rm rag-beginner-basics bash

# ── Cleanup ───────────────────────────────────────────────────────
clean:
	rm -rf vectordb_data/ __pycache__/ .pytest_cache/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

### Step 14: Verify your config files exist

```bash
ls -la pyproject.toml README.md .env.example .gitignore Makefile
```

---

## Phase 3: Install Dependencies

### Step 15: Create `rag_beginner_basics/__init__.py` (required before install)

Before installing dependencies, you **must** create `__init__.py` inside your package directory. Without it, hatchling (the build backend) cannot detect your Python package and `uv sync` will fail with:

```
ValueError: Unable to determine which files to ship inside the wheel
```

`mkdir` only creates empty directories — it does **not** create any files. A directory needs an `__init__.py` file to be recognized as a Python package.

```bash
touch rag_beginner_basics/__init__.py
```

This creates an empty `__init__.py` file. You'll add content to it later in Step 20.

### Step 16: Install all dependencies using uv

```bash
uv sync --all-extras
```

This will:
- Create a `.venv/` virtual environment automatically
- Install all core + dev dependencies
- Generate a `uv.lock` file

### Step 16b: Activate the virtual environment

`uv sync` creates a `.venv/` directory but does **not** activate it. Without activation, your `python` command still points to the system Python (e.g., `/opt/homebrew/opt/python@3.12/bin/python`), which won't have your project's dependencies installed.

```bash
source .venv/bin/activate
```

Verify it worked:

```bash
which python
```

You should see something like:

```
/Users/spothana/personal_projects/rag-beginner-basics/.venv/bin/python
```

If it still shows `/opt/homebrew/...`, the activation didn't work. Try closing and reopening your terminal, then run `source .venv/bin/activate` again.

> **Tip**: You'll need to run `source .venv/bin/activate` every time you open a new terminal. Alternatively, you can prefix any command with `uv run` (e.g., `uv run python -m rag_beginner_basics ingest`) which automatically uses the correct virtual environment without activation.

### Step 17: Verify the installation

```bash
# Check the virtual environment was created
ls -la .venv/

# Check installed packages
uv run python -c "import openai; print('openai:', openai.__version__)"
uv run python -c "import chromadb; print('chromadb:', chromadb.__version__)"
uv run python -c "import numpy; print('numpy:', numpy.__version__)"
uv run python -c "import faiss; print('faiss: OK')"
uv run python -c "import tiktoken; print('tiktoken: OK')"
```

---

## Phase 4: Create Sample Data

### Step 18: Create sample text files

Create 4 text files that your RAG pipeline will use as a knowledge base.

```bash
code data/sample_texts/artificial_intelligence.txt
```

Paste content about AI (write 400-800 words covering AI basics, history, types of AI, applications). Here is a starting point — **write your own version**:

```
Artificial Intelligence (AI) refers to the simulation of human intelligence
in machines that are programmed to think and learn like humans...

[Write about: definition, history, types (narrow AI, general AI),
applications (healthcare, finance, transportation), machine learning
as a subset, neural networks, current state of AI]
```

Repeat for 3 more files:

```bash
code data/sample_texts/python_programming.txt
```

(Write about Python: history, features, use cases, libraries, community)

```bash
code data/sample_texts/cloud_computing.txt
```

(Write about cloud: IaaS/PaaS/SaaS, providers, benefits, deployment models)

```bash
code data/sample_texts/machine_learning.txt
```

(Write about ML: supervised/unsupervised learning, algorithms, training, evaluation)

### Step 19: Verify your data files

```bash
ls -la data/sample_texts/
wc -w data/sample_texts/*.txt
```

The `wc -w` command counts words in each file. Aim for 400-800 words per file.

---

## Phase 5: Build the Python Package

You will create 7 Python files inside `rag_beginner_basics/`. Build them one at a time.

### Step 20: Add content to `rag_beginner_basics/__init__.py`

You created an empty `__init__.py` in Step 15. Now open it and add real content:

```bash
code rag_beginner_basics/__init__.py
```

Replace the empty file with:

```python
"""RAG Basics - A beginner-friendly RAG learning package."""

__version__ = "0.1.0"
```

### Step 21: Create `rag_beginner_basics/document_loader.py`

```bash
code rag_beginner_basics/document_loader.py
```

Write a module that:
- Defines a `Document` dataclass with `content: str` and `metadata: dict`
- Has a `load_text_file(path) -> Document` function
- Has a `load_pdf_file(path) -> Document` function (using `pypdf`)
- Has a `load_directory(path, glob_pattern) -> list[Document]` function

Hints:

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)

def load_text_file(file_path: str) -> Document:
    # Use Path(file_path).read_text() to read the file
    # Store filename and file size in metadata
    # Return a Document
    ...

def load_pdf_file(file_path: str) -> Document:
    # Use pypdf.PdfReader to extract text from each page
    # Join all pages into one string
    # Return a Document
    ...

def load_directory(dir_path: str, pattern: str = "*.txt") -> list[Document]:
    # Use Path(dir_path).glob(pattern) to find files
    # Call load_text_file for each .txt file
    # Call load_pdf_file for each .pdf file
    # Return the list
    ...
```

### Step 22: Test your document loader manually

```bash
uv run python -c "
from rag_beginner_basics.document_loader import load_text_file
doc = load_text_file('data/sample_texts/artificial_intelligence.txt')
print('Content length:', len(doc.content))
print('Metadata:', doc.metadata)
print('First 100 chars:', doc.content[:100])
"
```

### Step 23: Create `rag_beginner_basics/chunker.py`

```bash
code rag_beginner_basics/chunker.py
```

Write a module with 3 chunking strategies:

```python
from rag_beginner_basics.document_loader import Document

def chunk_by_size(document: Document, chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    # Split document.content into chunks of chunk_size characters
    # Each chunk overlaps with the previous by 'overlap' characters
    # Each chunk gets a copy of the original metadata + chunk_index
    ...

def chunk_by_sentences(document: Document, sentences_per_chunk: int = 5, overlap: int = 1) -> list[Document]:
    # Split on ". " to get sentences
    # Group sentences into chunks
    # Add overlap sentences between chunks
    ...

def chunk_recursive(document: Document, chunk_size: int = 500, separators: list[str] = None) -> list[Document]:
    # Try splitting by "\n\n" first, then "\n", then ". ", then " "
    # Recursively split until each piece is <= chunk_size
    ...
```

### Step 24: Test your chunker manually

```bash
uv run python -c "
from rag_beginner_basics.document_loader import load_text_file
from rag_beginner_basics.chunker import chunk_by_size

doc = load_text_file('data/sample_texts/artificial_intelligence.txt')
chunks = chunk_by_size(doc, chunk_size=200, overlap=20)
print(f'Document split into {len(chunks)} chunks')
for i, chunk in enumerate(chunks):
    print(f'  Chunk {i}: {len(chunk.content)} chars')
"
```

### Step 25: Create `rag_beginner_basics/embedder.py`

```bash
code rag_beginner_basics/embedder.py
```

Write:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dimensions = 1536

    def embed_text(self, text: str) -> list[float]:
        # Call self.client.embeddings.create(input=[text], model=self.model)
        # Return the embedding vector (response.data[0].embedding)
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Call embeddings.create with multiple texts at once
        # Return list of embedding vectors
        ...
```

### Step 26: Create `rag_beginner_basics/vector_store.py`

```bash
code rag_beginner_basics/vector_store.py
```

Write:

```python
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add(self, documents, embeddings, ids=None, metadatas=None):
        ...

    @abstractmethod
    def query(self, query_embedding, top_k=5, **kwargs):
        ...

    @abstractmethod
    def count(self) -> int:
        ...

class ChromaVectorStore(VectorStore):
    def __init__(self, collection_name="documents", persist_directory="./vectordb_data/chroma"):
        # Use chromadb.PersistentClient(path=persist_directory)
        # Get or create a collection with cosine similarity
        ...

    def add(self, documents, embeddings, ids=None, metadatas=None):
        # Generate IDs if not provided (use hashlib.md5)
        # Call self.collection.add(...)
        ...

    def query(self, query_embedding, top_k=5, **kwargs):
        # Call self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        # Return results dict with documents, distances, metadatas
        ...

    def count(self) -> int:
        # Return self.collection.count()
        ...

class FAISSVectorStore(VectorStore):
    def __init__(self, dimension=1536):
        # Use faiss.IndexFlatIP (inner product for cosine similarity)
        # Keep a parallel list (self.doc_store) for documents + metadata
        ...

    def add(self, documents, embeddings, ids=None, metadatas=None):
        # Normalize embeddings (L2 norm) for cosine similarity
        # Add to FAISS index
        # Append to doc_store
        ...

    def query(self, query_embedding, top_k=5, **kwargs):
        # Normalize query embedding
        # Search the FAISS index
        # Return matching documents with scores
        ...

    def count(self) -> int:
        # Return self.index.ntotal
        ...
```

### Step 27: Create `rag_beginner_basics/rag_pipeline.py`

```bash
code rag_beginner_basics/rag_pipeline.py
```

Write:

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
from rag_beginner_basics.document_loader import load_directory
from rag_beginner_basics.chunker import chunk_by_size
from rag_beginner_basics.embedder import OpenAIEmbedder
from rag_beginner_basics.vector_store import ChromaVectorStore

load_dotenv()

RAG_PROMPT_TEMPLATE = """Answer the question based on the context below.
If the context doesn't contain relevant information, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""

class RAGPipeline:
    def __init__(self, vector_store=None, embedder=None, llm_model="gpt-4o-mini"):
        self.embedder = embedder or OpenAIEmbedder()
        self.vector_store = vector_store or ChromaVectorStore()
        self.llm_model = llm_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def ingest(self, data_dir, pattern="*.txt", chunk_size=500, overlap=50):
        # 1. Load documents from data_dir
        # 2. Chunk each document
        # 3. Embed all chunks
        # 4. Add to vector store
        ...

    def query(self, question, top_k=5):
        # 1. Embed the question
        # 2. Query vector store for top_k similar chunks
        # 3. Build prompt with retrieved context
        # 4. Call OpenAI chat completion
        # 5. Return the answer
        ...
```

### Step 28: Create `rag_beginner_basics/main.py`

```bash
code rag_beginner_basics/main.py
```

Write:

```python
import argparse
from rag_beginner_basics.rag_pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="RAG Basics CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--data-dir", required=True, help="Path to data directory")
    ingest_parser.add_argument("--pattern", default="*.txt", help="File glob pattern")
    ingest_parser.add_argument("--chunk-size", type=int, default=500)
    ingest_parser.add_argument("--overlap", type=int, default=50)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG pipeline")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = RAGPipeline()
        pipeline.ingest(args.data_dir, args.pattern, args.chunk_size, args.overlap)
        print("Ingestion complete!")
    elif args.command == "query":
        pipeline = RAGPipeline()
        answer = pipeline.query(args.question, args.top_k)
        print(answer)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### Step 28b: Create `rag_beginner_basics/__main__.py`

This file is required so Python can run your package with `python -m rag_beginner_basics`. Without it, you'll get:

```
No module named rag_beginner_basics.__main__; 'rag_beginner_basics' is a package and cannot be directly executed
```

```bash
code rag_beginner_basics/__main__.py
```

Paste:

```python
"""Allow running the package with: python -m rag_beginner_basics"""

from rag_beginner_basics.main import main

main()
```

That's it — just 2 lines of real code. It imports `main()` from `main.py` and calls it. Think of it as a **doorbell** — `main.py` is the person inside the house, but `__main__.py` is what lets `python -m` ring the bell from outside.

### Step 29: Verify the package structure

```bash
find rag_beginner_basics/ -type f -name "*.py" | sort
```

Expected output:

```
rag_beginner_basics/__init__.py
rag_beginner_basics/__main__.py
rag_beginner_basics/chunker.py
rag_beginner_basics/document_loader.py
rag_beginner_basics/embedder.py
rag_beginner_basics/main.py
rag_beginner_basics/rag_pipeline.py
rag_beginner_basics/vector_store.py
```

---

## Phase 6: Write Unit Tests

### Step 30: Create `tests/__init__.py`

```bash
touch tests/__init__.py
```

### Step 31: Create `tests/test_document_loader.py`

```bash
code tests/test_document_loader.py
```

Write tests:

```python
from rag_beginner_basics.document_loader import Document, load_text_file, load_directory

def test_document_creation():
    doc = Document(content="Hello world", metadata={"source": "test"})
    assert doc.content == "Hello world"
    assert doc.metadata["source"] == "test"

def test_document_default_metadata():
    doc = Document(content="Hello")
    assert doc.metadata == {}

def test_load_text_file(tmp_path):
    # Create a temp file, write content, load it, assert content matches
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is test content.")
    doc = load_text_file(str(test_file))
    assert "test content" in doc.content

def test_load_directory(tmp_path):
    # Create 2 temp .txt files in tmp_path
    # Call load_directory(tmp_path)
    # Assert you get 2 documents back
    ...

def test_load_directory_empty(tmp_path):
    docs = load_directory(str(tmp_path))
    assert len(docs) == 0
```

### Step 32: Create `tests/test_chunker.py`

```bash
code tests/test_chunker.py
```

Write tests:

```python
from rag_beginner_basics.document_loader import Document
from rag_beginner_basics.chunker import chunk_by_size, chunk_by_sentences, chunk_recursive

def test_chunk_by_size_basic():
    doc = Document(content="a" * 1000, metadata={"source": "test"})
    chunks = chunk_by_size(doc, chunk_size=200, overlap=0)
    assert len(chunks) == 5  # 1000 / 200

def test_chunk_by_size_with_overlap():
    doc = Document(content="a" * 1000, metadata={"source": "test"})
    chunks = chunk_by_size(doc, chunk_size=200, overlap=50)
    assert len(chunks) > 5  # More chunks due to overlap

def test_chunk_by_size_preserves_metadata():
    doc = Document(content="a" * 500, metadata={"source": "test.txt"})
    chunks = chunk_by_size(doc, chunk_size=200, overlap=0)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"

def test_chunk_by_size_small_document():
    doc = Document(content="Short text", metadata={})
    chunks = chunk_by_size(doc, chunk_size=500)
    assert len(chunks) == 1

def test_chunk_by_sentences_basic():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five. Sentence six."
    doc = Document(content=text)
    chunks = chunk_by_sentences(doc, sentences_per_chunk=3, overlap=0)
    assert len(chunks) == 2

def test_chunk_recursive_basic():
    doc = Document(content="a" * 1000)
    chunks = chunk_recursive(doc, chunk_size=200)
    assert len(chunks) >= 5
    for chunk in chunks:
        assert len(chunk.content) <= 200
```

### Step 33: Create `tests/test_vector_store.py`

```bash
code tests/test_vector_store.py
```

Write tests that work **without an API key** (use random embeddings):

```python
import numpy as np
from rag_beginner_basics.vector_store import FAISSVectorStore

def test_faiss_add_and_count():
    store = FAISSVectorStore(dimension=128)
    embeddings = np.random.rand(3, 128).astype("float32").tolist()
    store.add(
        documents=["doc1", "doc2", "doc3"],
        embeddings=embeddings,
    )
    assert store.count() == 3

def test_faiss_query():
    store = FAISSVectorStore(dimension=128)
    embeddings = np.random.rand(5, 128).astype("float32").tolist()
    store.add(
        documents=["apple", "banana", "cherry", "date", "elderberry"],
        embeddings=embeddings,
    )
    results = store.query(query_embedding=embeddings[0], top_k=3)
    assert len(results["documents"]) == 3

def test_faiss_empty_store():
    store = FAISSVectorStore(dimension=128)
    assert store.count() == 0

# Add more tests: query with filter, persistence (save/load), etc.
```

---

## Phase 7: Run Tests and Lint

### Step 34: Run unit tests

```bash
uv run pytest -v
```

If tests fail, read the error messages carefully and fix your code. Common issues:
- Import errors → check file names and function names match
- `AttributeError` → check your class/function signatures
- `AssertionError` → check your logic

### Step 35: Run linter

```bash
uv run ruff check .
```

If there are errors, auto-fix what you can:

```bash
uv run ruff check . --fix
```

### Step 36: Format code

```bash
uv run ruff format .
```

### Step 37: Run tests again to make sure formatting didn't break anything

```bash
uv run pytest -v
```

### Step 38: Use the Makefile shortcuts

Now try the same things via Makefile:

```bash
make test
make lint
make format
```

---

## Phase 8: Test the CLI

### Step 39: Set up your API key

```bash
cp .env.example .env
code .env
```

Replace `sk-your-api-key-here` with your actual OpenAI API key.

### Step 40: Test the CLI ingest command

```bash
uv run rag-beginner-basics ingest --data-dir ./data/sample_texts/
```

### Step 41: Test the CLI query command

```bash
uv run rag-beginner-basics query "What is machine learning?"
```

### Step 42: Try different queries

```bash
uv run rag-beginner-basics query "What are the benefits of cloud computing?"
uv run rag-beginner-basics query "Why is Python popular?"
uv run rag-beginner-basics query "What is the difference between AI and ML?"
```

---

## Final Verification Checklist

Run through this checklist to confirm everything works:

```bash
# 1. Dependencies install cleanly
make setup

# 2. All tests pass
make test

# 3. No lint errors
make lint

# 4. Code is formatted
make format

# 5. CLI help works
uv run rag-beginner-basics --help
uv run rag-beginner-basics ingest --help
uv run rag-beginner-basics query --help

# 6. Package imports work
uv run python -c "from rag_beginner_basics import __version__; print(__version__)"
uv run python -c "from rag_beginner_basics.document_loader import Document; print('OK')"
uv run python -c "from rag_beginner_basics.chunker import chunk_by_size; print('OK')"
uv run python -c "from rag_beginner_basics.vector_store import FAISSVectorStore; print('OK')"

```

---

## Summary of All Commands Used

| Phase | Commands |
|-------|----------|
| Prerequisites | `python3 --version`, `brew install python@3.12`, `curl ... uv` |
| Project setup | `mkdir -p`, `cd` |
| Config files | `code pyproject.toml`, `code .gitignore`, `code Makefile`, `code .env.example` |
| Dependencies | `uv sync --all-extras` |
| Manual testing | `uv run python -c "..."` |
| Unit tests | `uv run pytest -v`, `make test` |
| Linting | `uv run ruff check .`, `uv run ruff check . --fix`, `uv run ruff format .` |
| CLI | `uv run rag-beginner-basics ingest ...`, `uv run rag-beginner-basics query ...` |
| Makefile | `make setup`, `make test`, `make lint`, `make format` |
