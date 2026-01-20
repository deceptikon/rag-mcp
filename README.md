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
3.  **An Ollama Embedding Model:** You will need an embedding model to generate the vectors. We recommend `nomic-embed-text`. You can pull it with the following command:
    ```bash
    ollama pull nomic-embed-text
    ```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd rag-mcp
    ```

2.  **Install the required Python packages using uv:**
    ```bash
    uv sync
    ```

## Usage

The primary commands for `rag-mcp` are `index` and `search`.

### Indexing a Project

To index a project, run the following command using `uv`:

```bash
uv run main.py index /path/to/your/project --model nomic-embed-text
```

**Arguments:**

*   `project_path` (required): The path to the project directory you want to index.
*   `--model` (optional, default: `nomic-embed-text`): The name of the Ollama embedding model to use. This model must be available in your local Ollama instance.

### Searching a Project

To search an indexed project, run the following command:

```bash
uv run main.py search /path/to/your/project "your search query" --model nomic-embed-text
```

**Arguments:**

*   `project_path` (required): The path to the project directory you want to search in.
*   `query` (required): The search query.
*   `--model` (optional, default: `nomic-embed-text`): The name of the Ollama embedding model to use.
*   `--top_k` (optional, default: `5`): The number of results to return.

## Global Alias

To make the `rag-mcp` tool easily accessible from anywhere, you can use the included `mcp` wrapper script.

1.  **Make sure the script is executable:**
    ```bash
    chmod +x mcp
    ```

2.  **Create a symbolic link** to the `mcp` script in a directory that is in your system's `PATH`. A common choice is `/usr/local/bin`.

    ```bash
    sudo ln -s /path/to/your/rag-mcp/mcp /usr/local/bin/mcp
    ```
    *Replace `/path/to/your/rag-mcp` with the absolute path to the `rag-mcp` directory.*

3.  **Verify the installation** by running:
    ```bash
    mcp --help
    ```

Now you can use `mcp` as a global command:

```bash
mcp index /path/to/your/project
mcp search /path/to/your/project "your search query"
```

## How it Works
1.  **Scanning and Chunking:** The tool recursively scans the specified project directory. It uses logic adapted from `RAG/scanner.py` to intelligently chunk files based on their type (e.g., Python, JavaScript, Markdown), while ignoring irrelevant files and directories (like `node_modules`, `.git`, etc.).
2.  **Embedding:** For each chunk, the tool makes a request to your local Ollama API to generate a vector embedding.
3.  **Storage:** The chunk's content, metadata (e.g., source file path), and the generated embedding are stored together in a local ChromaDB database.

## Future Development

*   **`ctx` Integration:** Deeper integration with the `ctx` tool to expose search results within your existing workflow.
*   **Web UI:** A simple web interface for visualizing and interacting with the context.
