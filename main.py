import argparse
import os
from app.chunker import load_and_chunk_project
from app.vector_store import VectorStoreManager

def main():
    parser = argparse.ArgumentParser(description="rag-mcp: Codebase-Aware RAG Context Builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a project directory.")
    index_parser.add_argument("project_path", help="The path to the project directory to index.")
    index_parser.add_argument("--model", default="mxbai-embed-large", help="The name of the Ollama embedding model to use.")

    args = parser.parse_args()

    if args.command == "index":
        print(f"Indexing project at: {args.project_path}")
        print(f"Using model: {args.model}")

        # Get collection name from project path
        project_name = os.path.basename(os.path.abspath(args.project_path))
        collection_name = f"rag-mcp-{project_name}"

        # Chunk the project
        documents = load_and_chunk_project(args.project_path)

        # Create a vector store manager and add the documents
        vector_store = VectorStoreManager(db_path=".rag_db", collection_name=collection_name)
        vector_store.add_documents(documents, model=args.model)

        print("Indexing complete.")
        stats = vector_store.get_stats()
        print(f"Total items in '{stats['collection_name']}': {stats['total_items']}")

if __name__ == "__main__":
    main()