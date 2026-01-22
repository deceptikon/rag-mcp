import os
import logging
from typing import Optional
from mcp.server.fastmcp import FastMCP
from app.chunker import load_and_chunk_project
from app.vector_store import VectorStoreManager

# Setup Logging to be actually visible
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vox-brain")

# Initialize FastMCP Server
mcp = FastMCP("VOX Brain (RAG)", dependencies=["chromadb", "langchain-text-splitters"])


def get_collection_name(project_path: str) -> str:
    """Gets the collection name from the project path."""
    project_name = os.path.basename(os.path.abspath(project_path))
    return f"rag-mcp-{project_name}"


@mcp.tool()
async def index_project(project_path: str, model: str = "nomic-embed-text") -> str:
    """
    Indexes a project directory into the vector store.
    Args:
        project_path: Absolute path to the project root.
        model: The embedding model to use.
    """
    logger.info(f"🚀 Starting indexing for: {project_path}")

    if not os.path.isdir(project_path):
        logger.error(f"❌ Invalid path: {project_path}")
        return f"Error: {project_path} is not a valid directory."

    try:
        collection_name = get_collection_name(project_path)
        logger.info(f"📦 Loading and chunking files...")
        documents = load_and_chunk_project(project_path)

        logger.info(
            f"💾 Upserting {len(documents)} chunks to collection: {collection_name}"
        )
        vector_store = VectorStoreManager(
            db_path=".rag_db", collection_name=collection_name
        )
        vector_store.add_documents(documents, model=model)

        logger.info(f"✅ Indexing complete for {project_path}")
        return f"Successfully indexed {len(documents)} chunks into {collection_name}."
    except Exception as e:
        logger.exception("🔥 Indexing failed")
        return f"Indexing failed: {str(e)}"


@mcp.tool()
async def search_project(project_path: str, query: str, top_k: int = 5) -> str:
    """
    Searches the project context for relevant code blocks.
    Args:
        project_path: Absolute path to the project root.
        query: The semantic search query.
        top_k: Number of results to return.
    """
    logger.info(f"🔍 Searching in {project_path} for: '{query}'")

    collection_name = get_collection_name(project_path)
    vector_store = VectorStoreManager(
        db_path=".rag_db", collection_name=collection_name
    )

    try:
        results = vector_store.search(
            query_text=query, model="nomic-embed-text", n_results=top_k
        )

        output = []
        for res in results:
            output.append(
                f"--- SOURCE: {res.get('source', 'unknown')} (Relevance: {res.get('relevance', 'N/A')}) ---\n{res.get('content', '')}"
            )

        logger.info(f"✨ Found {len(results)} relevant results.")
        return "\n\n".join(output) if output else "No relevant results found."
    except Exception as e:
        logger.error(f"⚠️ Search failed: {str(e)}")
        return f"Search error: {str(e)}"


if __name__ == "__main__":
    # When run directly, use Stdio transport (default for FastMCP)
    mcp.run()
