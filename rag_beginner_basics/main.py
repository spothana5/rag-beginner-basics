"""CLI entry point for the RAG Basics pipeline."""

import argparse
import sys

from dotenv import load_dotenv

from rag_beginner_basics.rag_pipeline import RAGPipeline
from rag_beginner_basics.vector_store import ChromaVectorStore


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG Basics - Learn RAG by doing")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument("--data-dir", required=True, help="Directory containing documents")
    ingest_parser.add_argument("--pattern", default="*.txt", help="File pattern to match (default: *.txt)")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default: 500)")
    ingest_parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap in characters (default: 50)")
    ingest_parser.add_argument("--collection", default="rag-basics", help="ChromaDB collection name")
    ingest_parser.add_argument("--persist-dir", default="./vectordb_data/chromadb", help="ChromaDB persist directory")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question using RAG")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    query_parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model for generation")
    query_parser.add_argument("--collection", default="rag-basics", help="ChromaDB collection name")
    query_parser.add_argument("--persist-dir", default="./vectordb_data/chromadb", help="ChromaDB persist directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        store = ChromaVectorStore(collection_name=args.collection, persist_dir=args.persist_dir)
        pipeline = RAGPipeline(
            vector_store=store,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )
        count = pipeline.ingest(args.data_dir, pattern=args.pattern)
        print(f"Ingested {count} chunks into collection '{args.collection}'")

    elif args.command == "query":
        store = ChromaVectorStore(collection_name=args.collection, persist_dir=args.persist_dir)
        pipeline = RAGPipeline(
            vector_store=store,
            llm_model=args.model,
        )
        result = pipeline.query(args.question, top_k=args.top_k)

        print(f"\nQuestion: {result['question']}\n")
        print(f"Answer: {result['answer']}\n")
        print(f"Sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
