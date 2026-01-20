# rag-mcp: Codebase-Aware RAG Context Builder

`rag-mcp` is a command-line tool for creating a local Retrieval-Augmented Generation (RAG) context from your project's codebase. It scans your project, chunks the code and documentation, and then uses a local Ollama embedding model to populate a vector database.

This tool is designed to be an extension of the existing `ctx` workflow, allowing you to build a powerful semantic search capability for your projects without disrupting your current development process.

## Features

*   **Code-Aware Chunking:** Intelligently chunks code and documentation based on file type.
*   **Local First:** Uses your own local Ollama instance for embedding generation. No data leaves your machine.
*   **Vector Database:** Stores embeddings and associated metadata (file path, content) in a local [ChromaDB](https://www.trychroma.com/) database (which uses SQLite).
*   **Simple CLI:** A straightforward command-line interface for indexing your projects.

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Ollama:** Follow the instructions at [ollama.ai](https://ollama.ai/) to install and run Ollama on your system.
3.  **An Ollama Embedding Model:** You will need an embedding model to generate the vectors. We recommend `mxbai-embed-large`. You can pull it with the following command:
    ```bash
    ollama pull mxbai-embed-large
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rag-mcp
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file will be created in a subsequent step).*

## Usage

The primary command for `rag-mcp` is `index`, which scans a directory and populates the vector database.

### Indexing a Project

To index a project, run the following command:

```bash
python main.py index /path/to/your/project --model mxbai-embed-large
```

**Arguments:**

*   `<project_path>` (required): The path to the project directory you want to index.
*   `--model` (optional, default: `mxbai-embed-large`): The name of the Ollama embedding model to use. This model must be available in your local Ollama instance.

### Database Location

The vector database will be stored in a `.rag_db` directory within the `rag-mcp` project folder.

## How it Works

1.  **Scanning and Chunking:** The tool recursively scans the specified project directory. It uses logic adapted from `RAG/scanner.py` to intelligently chunk files based on their type (e.g., Python, JavaScript, Markdown), while ignoring irrelevant files and directories (like `node_modules`, `.git`, etc.).
2.  **Embedding:** For each chunk, the tool makes a request to your local Ollama API to generate a vector embedding.
3.  **Storage:** The chunk's content, metadata (e.g., source file path), and the generated embedding are stored together in a local ChromaDB database.

## Future Development

*   **Search Command:** A `search` command to perform semantic search over the indexed database.
*   **`ctx` Integration:** Deeper integration with the `ctx` tool to expose search results within your existing workflow.
*   **Web UI:** A simple web interface for visualizing and interacting with the context.
