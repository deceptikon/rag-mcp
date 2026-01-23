# 🧠 VOX Brain (MCP Server)

A Model Context Protocol (MCP) server that provides RAG (Retrieval Augmented Generation) and Context capabilities for OpenCode agents.

## Features

- **Vector Search:** Indexes codebase using `chromadb` and `nomic-embed-text`.
- **RAG Query:** Answering questions about the codebase using `qwen2.5-coder`.
- **Context Resources:** Exposes project rules, documentation, and file structure via MCP resources.

## Usage

This server is managed automatically by the `vox` CLI tool.

### 1. Setup
First, register your project to get a **Project ID**:
```bash
vox init /path/to/your/project "Project Name"
# Output: Returns a hash ID (e.g., cdb4ee2a)
```

List all registered projects:
```bash
vox list
```

### 2. Indexing (Memory)
Index the codebase into the Vector Brain:
```bash
vox sync-v <project_id>
```

### 3. Interaction
Search for code by meaning:
```bash
vox search-v <project_id> "how does authentication work?"
```

Ask the AI to explain something:
```bash
vox ask-v <project_id> "Explain the main logic in server.py"
```

### 4. Rules & Docs
Add context that isn't in the code (stored as JSONL resources):
```bash
vox add-doc <project_id> rule "Always use async/await" "Async Standard"
vox list-docs <project_id>
```

## Architecture

- `server.py`: Main FastMCP entry point.
- `app/chunker.py`: Intelligent code chunking logic.
- `app/vector_store.py`: ChromaDB management.

## Installation

Installed automatically via `ctx-core/install.sh`.
