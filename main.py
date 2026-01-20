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
    index_parser.add_argument("--model", default="nomic-embed-text", help="The name of the Ollama embedding model to use.")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the vector store for a given query.")
    search_parser.add_argument("project_path", help="The path to the project directory to search in.")
    search_parser.add_argument("query", help="The search query.")
    search_parser.add_argument("--model", default="nomic-embed-text", help="The name of the Ollama embedding model to use.")
    search_parser.add_argument("--top_k", type=int, default=5, help="The number of results to return.")

    args = parser.parse_args()

    # Get collection name from project path for both commands
    project_name = os.path.basename(os.path.abspath(args.project_path))
    collection_name = f"rag-mcp-{project_name}"

    if args.command == "index":
        print(f"Indexing project at: {args.project_path}")
        print(f"Using model: {args.model}")

        # Chunk the project
        documents = load_and_chunk_project(args.project_path)

        # Create a vector store manager and add the documents
        vector_store = VectorStoreManager(db_path=".rag_db", collection_name=collection_name)
        vector_store.add_documents(documents, model=args.model)

        print("Indexing complete.")
        stats = vector_store.get_stats()
        print(f"Total items in '{stats['collection_name']}': {stats['total_items']}")

    elif args.command == "search":
        print(f"Searching in project: {args.project_path}")
        print(f"Query: '{args.query}'")
        
        # Create a vector store manager
        vector_store = VectorStoreManager(db_path=".rag_db", collection_name=collection_name)

        # Search the vector store
        results = vector_store.search(query_text=args.query, model=args.model, n_results=args.top_k)

        # Print the results
        if results:
            print("\n--- Search Results ---")
            for i, result in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Source: {result['source']}")
                print(f"  Relevance: {result['relevance']:.4f}")
                print(f"  Content: {result['content'][:500]}...")
                print("-" * 20)
        else:
            print("No results found.")

if __name__ == "__main__":
    main()