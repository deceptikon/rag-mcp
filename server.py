import sys
import os
import logging
import json
import ollama
from typing import Optional
from mcp.server.fastmcp import FastMCP
from app.chunker import load_and_chunk_project
from app.vector_store import VectorStoreManager
from sentence_transformers import CrossEncoder
import numpy as np

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
        db_path="$project_path/.rag_db"
        vector_store = VectorStoreManager(
            db_path=db_path, collection_name=collection_name
        )
        vector_store.add_documents(documents, model=model)

        logger.info(f"✅ Indexing complete for {project_path}")
        sys.stdout.flush()
        return f"Successfully indexed {len(documents)} chunks into {collection_name}."
    except Exception as e:
        logger.exception("🔥 Indexing failed")
        return f"Indexing failed: {str(e)}"


# Инициализируем реранкер (лучше вынести в глобальную область или синглтон)
# Модель ms-marco-MiniLM-L-6-v2 — золотой стандарт: быстрая и точная.
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#async def rewrite_query_for_code(query: str) -> str:
#    prompt = f"Given the programming question: '{query}', list 3-5 technical keywords or function names that might appear in the source code. Output only keywords separated by commas."
#    # Вызываем Ollama gemma3:4b-it-qat (она очень быстрая, это займет <1 сек)
#
#    response = ollama.chat(
#        model=model,
#        messages=[
#            {"role": "system", "content": system_prompt},
#            {"role": "user", "content": user_prompt},
#        ],
#        options={
#            "temperature": 0.0,  # Делает ответы стабильными и точными
#            "num_ctx": 8192,     # Увеличиваем окно, чтобы влезло больше файлов (по умолчанию 2048)
#            "num_predict": 1024  # Ограничиваем длину ответа, чтобы экономить ресурсы
#        }
#    )
#    keywords = response["message"]["content"]    # Результат: "AuthService, JWT, login, authenticate, token"
#    return keywords

async def rewrite_query_for_code(query: str) -> str:
    # Используем максимально сжатый системный промпт
    system_msg = "You are a technical search assistant. Output ONLY a comma-separated list of technical terms. No prose."
    user_msg = f"Translate this intent into code keywords: '{query}'"

    try:
        response = ollama.chat(
            model="gemma3:4b-it-qat",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options={
                "temperature": 0.1,    # Минимум креатива
                "num_predict": 30,     # Нам нужно всего пару слов, не даем модели рассуждать
                "stop": ["\n", "Sure", "Here"] # Обрезаем лишнюю вежливость
            }
        )
        keywords = response["message"]["content"].strip()
        # Очистка от возможного мусора (кавычки, точки в конце)
        return keywords.replace('"', '').replace('.', '')
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return "" # Если упало, поиск пойдет просто по оригинальному запросу

@mcp.tool()
async def search_project(project_path: str, query: str, top_k: int = 20) -> str:
    """
    Улучшенный поиск с реранжированием и структурированием для Gemma 3.
    """
    # 1. Query Expansion (НЕ заменяем, а расширяем)
    # Генерируем ключевые слова, которые могут быть в коде
    search_terms = await rewrite_query_for_code(query)

    # Объединяем: оригинальный вопрос + технические термины
    # Это гарантирует, что мы ищем и по смыслу, и по ключевым словам
    combined_query = f"{query} {search_terms}"

    logger.info(f"🔎 Combined Query: {combined_query}")

    collection_name = get_collection_name(project_path)
    vector_store = VectorStoreManager(
        db_path=".rag_db", collection_name=collection_name
    )

    try:
        # 1. Retrieval: Берем с запасом (20 результатов), чтобы было из чего выбирать
        initial_results = vector_store.search(
            query_text=combined_query,  # Ищем по расширенному запросу
            model="nomic-embed-text",
            n_results=20
        )

        if not initial_results:
            return "No relevant results found."

        # 2. Reranking: Сравниваем запрос с каждым найденным куском кода
        # Формируем пары [вопрос, код] для оценки
        pairs = [[query, res.get('content', '')] for res in initial_results]
        scores = reranker.predict(pairs)

        # Добавляем скоры в результаты и сортируем
        for i, res in enumerate(initial_results):
            res['rerank_score'] = scores[i]

        # Сортируем по убыванию релевантности и берем нужные top_k
        reranked_results = sorted(
            initial_results, key=lambda x: x['rerank_score'], reverse=True
        )[:top_k]

        # 3. Formatting: Собираем контекст в XML-подобную структуру
        formatted_output = ["<context>"]

        for i, res in enumerate(reranked_results):
            source = res.get('source', 'unknown')
            content = res.get('content', '')
            # Четко отделяем каждый документ
            doc_block = (
                f"### DOCUMENT {i+1}\n"
                f"FILE_PATH: {source}\n"
                f"CODE_CONTENT:\n{content}\n"
                f"--- END OF DOCUMENT {i+1} ---"
            )
            formatted_output.append(doc_block)

        formatted_output.append("</context>")

        logger.info(f"✨ Reranking complete. Best score: {reranked_results[0].get('rerank_score'):.4f}")
        return "\n\n".join(formatted_output)

    except Exception as e:
        logger.error(f"⚠️ Search/Rerank failed: {str(e)}")
        return f"Search error: {str(e)}"



@mcp.tool()
async def ask_project(
    project_path: str, question: str, model: str = "gemma3:4b-it-qat"
) -> str:
    """
    Asks a question about the project, using retrieved context to answer.
    Args:
        project_path: Absolute path to the project root.
        question: The user's question.
        model: The Ollama model to use for generation (default: gemma3:4b-it-qat).
    """
    logger.info(f"🤔 Asking '{model}' about {project_path}: '{question}'")

    # 1. Reuse search logic to get context
    context = await search_project(project_path, question)

    if context.startswith("Search error") or context == "No relevant results found.":
        return f"Could not retrieve context to answer the question. Reason: {context}"

    # 2. Construct Prompt
#    system_prompt = (
#        "You are an expert software engineer assistant. "
#        "Answer the user's question based strictly on the provided code context. "
#        "If the answer is not in the context, state that you don't know."
#    )
    system_prompt = (
        "You are an expert software engineer assistant specializing in RAG systems. "
        "Your task is to answer the user's question based ONLY on the provided code context. "
        "For every piece of information you provide, you MUST cite the source file using the format [SOURCE: filename]. "
    )

    #user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    user_prompt = f"""
    ### SOURCE CODE CONTEXT
    --------------------------------------------------
    {context}
    --------------------------------------------------

    ### USER QUESTION
    {question}

    ### ANSWER
    """

    user_prompt += (
        "#### Response Instructions:\n"
        "1. Analyze the context provided above and identify the key snippets that directly relate to the question.\n"
        "2. Provide a detailed answer using ONLY information from these snippets.\n"
        "3. Cite the source file for each key assertion using the format [SOURCE: filename].\n"
        "4. If information is missing, EXPLICITLY reply that you refuse to answer the specific part that is missing context.\n\n"
        "#### Answer:"
    )

    try:
        # 3. Call Ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.0,  # Делает ответы стабильными и точными
                "num_ctx": 8192,     # Увеличиваем окно, чтобы влезло больше файлов (по умолчанию 2048)
                "num_predict": 1024  # Ограничиваем длину ответа, чтобы экономить ресурсы
            }
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
    docs_path = os.path.expanduser(f"$VOX_HOME/context/docs/{project_id}/docs.jsonl")
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
    docs_path = os.path.expanduser(f"$VOX_HOME/context/docs/{project_id}/docs.jsonl")
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
    docs_path = os.path.expanduser(f"$VOX_HOME/context/docs/{project_id}/docs.jsonl")
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
    # We can use the $VOX_HOME/context/projects/{id}/config.json file
    config_path = os.path.expanduser(
        f"$VOX_HOME/context/projects/{project_id}/config.json"
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
