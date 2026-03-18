"""End-to-end RAG pipeline combining retrieval and generation."""

from openai import OpenAI

from rag_beginner_basics.chunker import chunk_by_size
from rag_beginner_basics.document_loader import load_directory
from rag_beginner_basics.embedder import OpenAIEmbedder
from rag_beginner_basics.vector_store import ChromaVectorStore, VectorStore

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question based on the provided context.
If the context does not contain enough information to answer, say so honestly.

## Context
{context}

## Question
{question}

## Instructions
- Answer based ONLY on the context provided above
- If the context doesn't contain the answer, say "I don't have enough information to answer this."
- Cite which source document(s) the information comes from
- Be concise and direct
"""


class RAGPipeline:
    """A complete RAG pipeline: ingest documents, then query with retrieval-augmented generation."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedder: OpenAIEmbedder | None = None,
        llm_model: str = "gpt-4o-mini",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embedder = embedder or OpenAIEmbedder()
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_client = OpenAI()

    def ingest(self, data_dir: str, pattern: str = "*.txt") -> int:
        """Load, chunk, embed, and store documents from a directory."""
        documents = load_directory(data_dir, pattern=pattern)

        all_chunks = []
        for doc in documents:
            chunks = chunk_by_size(doc, chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.content for c in all_chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.vector_store.add(all_chunks, embeddings)

        return len(all_chunks)

    def query(self, question: str, top_k: int = 3) -> dict:
        """Run the full RAG pipeline: retrieve context, build prompt, generate answer."""
        # Retrieve
        query_embedding = self.embedder.embed_text(question)
        retrieved = self.vector_store.query(query_embedding, top_k=top_k)

        # Build prompt
        context_parts = []
        for chunk in retrieved:
            source = chunk.get("metadata", {}).get("source", "unknown")
            context_parts.append(f"[Source: {source}]\n{chunk['content']}")

        context = "\n\n---\n\n".join(context_parts)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

        # Generate
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        answer = response.choices[0].message.content

        return {
            "question": question,
            "answer": answer,
            "sources": [c.get("metadata", {}).get("source", "unknown") for c in retrieved],
            "num_chunks_retrieved": len(retrieved),
        }
