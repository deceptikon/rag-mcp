import sys
import os
import logging
import json
import ollama
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
        sys.stdout.flush()
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


@mcp.tool()
async def ask_project(
    project_path: str, question: str, model: str = "qwen2.5-coder:1.5b-base"
) -> str:
    """
    Asks a question about the project, using retrieved context to answer.
    Args:
        project_path: Absolute path to the project root.
        question: The user's question.
        model: The Ollama model to use for generation (default: qwen2.5-coder:1.5b-base).
    """
    logger.info(f"🤔 Asking '{model}' about {project_path}: '{question}'")

    # 1. Reuse search logic to get context
    context = await search_project(project_path, question, top_k=5)

    if context.startswith("Search error") or context == "No relevant results found.":
        return f"Could not retrieve context to answer the question. Reason: {context}"

    # 2. Construct Prompt
    system_prompt = (
        "You are an expert software engineer assistant. "
        "Answer the user's question based strictly on the provided code context. "
        "If the answer is not in the context, state that you don't know."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        # 3. Call Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = response["message"]["content"]
        logger.info("💡 Answer generated successfully")
        return answer
    except Exception as e:
        logger.exception("🔥 LLM generation failed")
        return f"Error generating answer: {str(e)}"


@mcp.resource("vox://{project_id}/rules")
def get_project_rules(project_id: str) -> str:
    """Reads the project rules from the context store."""
    docs_path = os.path.expanduser(f"~/.opencode/context/docs/{project_id}/docs.jsonl")
    logger.info(f"📜 Reading rules from: {docs_path}")

    if not os.path.exists(docs_path):
        return "No rules found for this project."

    rules = []
    try:
        with open(docs_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "rule":
                        rules.append(f"### {data.get('title')}\n{data.get('content')}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"Error reading rules: {str(e)}"

    return "\n\n".join(rules) if rules else "No rules found."


@mcp.resource("vox://{project_id}/docs")
def get_project_docs(project_id: str) -> str:
    """Reads the project documentation from the context store."""
    docs_path = os.path.expanduser(f"~/.opencode/context/docs/{project_id}/docs.jsonl")
    logger.info(f"📜 Reading docs from: {docs_path}")

    if not os.path.exists(docs_path):
        return "No documentation found for this project."

    docs = []
    try:
        with open(docs_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "doc":
                        docs.append(f"### {data.get('title')}\n{data.get('content')}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"Error reading docs: {str(e)}"

    return "\n\n".join(docs) if docs else "No documentation found."


@mcp.resource("vox://{project_id}/notes")
def get_project_notes(project_id: str) -> str:
    """Reads the project notes from the context store."""
    docs_path = os.path.expanduser(f"~/.opencode/context/docs/{project_id}/docs.jsonl")
    logger.info(f"📜 Reading notes from: {docs_path}")

    if not os.path.exists(docs_path):
        return "No notes found for this project."

    notes = []
    try:
        with open(docs_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("type") == "note":
                        notes.append(f"### {data.get('title')}\n{data.get('content')}")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return f"Error reading notes: {str(e)}"

    return "\n\n".join(notes) if notes else "No notes found."


@mcp.resource("vox://{project_id}/tree")
def get_project_tree(project_id: str) -> str:
    """Gets the file tree structure of the project."""
    # We need to resolve the project_id to a path first.
    # We can use the ~/.opencode/context/projects/{id}/config.json file
    config_path = os.path.expanduser(
        f"~/.opencode/context/projects/{project_id}/config.json"
    )

    if not os.path.exists(config_path):
        return f"Project {project_id} not found."

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            project_path = config.get("path")
    except Exception as e:
        return f"Error reading project config: {str(e)}"

    if not project_path or not os.path.exists(project_path):
        return f"Project path not found: {project_path}"

    logger.info(f"🌳 Generating tree for: {project_path}")

    tree_lines = []
    try:
        for root, dirs, files in os.walk(project_path):
            # Filter excluded directories (basic)
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    "node_modules",
                    "__pycache__",
                    ".venv",
                    "venv",
                    "dist",
                    "build",
                }
            ]

            level = root.replace(project_path, "").count(os.sep)
            indent = " " * 4 * (level)
            tree_lines.append(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 4 * (level + 1)
            for f in files:
                if not f.startswith("."):  # Skip hidden files
                    tree_lines.append(f"{subindent}{f}")

        return "\n".join(tree_lines)
    except Exception as e:
        return f"Error generating tree: {str(e)}"


if __name__ == "__main__":
    # When run directly, use Stdio transport (default for FastMCP)
    mcp.run()
