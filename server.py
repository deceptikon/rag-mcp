from flask import Flask, request, jsonify
import os
import threading
from app.chunker import load_and_chunk_project
from app.vector_store import VectorStoreManager

app = Flask(__name__)

def get_collection_name(project_path: str) -> str:
    """Gets the collection name from the project path."""
    project_name = os.path.basename(os.path.abspath(project_path))
    return f"rag-mcp-{project_name}"

@app.route('/status', methods=['GET'])
def status():
    """Returns the status of the server."""
    return jsonify({"status": "ok"})

@app.route('/index', methods=['POST'])
def index_project():
    """Indexes a project directory."""
    data = request.get_json()
    project_path = data.get('project_path')
    model = data.get('model', 'nomic-embed-text')

    if not project_path or not os.path.isdir(project_path):
        return jsonify({"error": "Invalid 'project_path' provided."}), 400

    collection_name = get_collection_name(project_path)

    def index_in_background():
        print(f"Indexing project at: {project_path}")
        documents = load_and_chunk_project(project_path)
        vector_store = VectorStoreManager(db_path=".rag_db", collection_name=collection_name)
        vector_store.add_documents(documents, model=model)
        print("Indexing complete.")

    # Run indexing in a background thread to avoid blocking the request
    thread = threading.Thread(target=index_in_background)
    thread.start()

    return jsonify({
        "message": "Indexing started in the background.",
        "collection_name": collection_name
    })

@app.route('/search', methods=['POST'])
def search_project():
    """Searches the vector store for a given query."""
    data = request.get_json()
    project_path = data.get('project_path')
    query = data.get('query')
    model = data.get('model', 'nomic-embed-text')
    top_k = data.get('top_k', 5)

    if not project_path or not query:
        return jsonify({"error": "Missing 'project_path' or 'query'."}), 400

    collection_name = get_collection_name(project_path)
    vector_store = VectorStoreManager(db_path=".rag_db", collection_name=collection_name)
    
    try:
        results = vector_store.search(query_text=query, model=model, n_results=top_k)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_server():
    """Runs the Flask server using waitress."""
    from waitress import serve
    print("Starting Flask server on http://localhost:8088")
    serve(app, host="0.0.0.0", port=8088)

if __name__ == '__main__':
    run_server()
