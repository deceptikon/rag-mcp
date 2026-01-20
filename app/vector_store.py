import sys
import json
import requests
from pathlib import Path

def get_ollama_embedding(model_name, prompt):
    """
    Gets an embedding from the Ollama API.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({
                "model": model_name,
                "prompt": prompt
            })
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting Ollama embedding: {e}")
        print("Please ensure Ollama is running and the model is available.")
        sys.exit(1)


class VectorStoreManager:
    """Manages ChromaDB integration."""

    def __init__(self, db_path: str, collection_name: str = "default"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        try:
            import chromadb
        except ImportError:
            print("ChromaDB not installed. Run: pip install chromadb")
            sys.exit(1)

        self._chroma_client = chromadb.PersistentClient(path=str(self.db_path))
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name
        )

    def add_documents(self, documents: list, model: str):
        """
        Adds a list of documents to the vector store.
        """
        print(f"Adding {len(documents)} documents to the vector store...")

        # Prepare documents for ChromaDB
        doc_contents = []
        metadatas = []
        ids = []
        embeddings = []

        for i, doc in enumerate(documents):
            doc_contents.append(doc.page_content)
            metadatas.append(doc.metadata)
            ids.append(f"doc_{i}") # Simple ID for now
            
            # Get embedding from Ollama
            embedding = get_ollama_embedding(model, doc.page_content)
            embeddings.append(embedding)

            print(f"  Embedded document {i+1}/{len(documents)}")

        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(doc_contents), batch_size):
            batch_docs = doc_contents[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            self._collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
            print(
                f"  Indexed {min(i + batch_size, len(doc_contents))}/{len(doc_contents)} items"
            )

        print(f"Indexed {len(doc_contents)} items into ChromaDB")

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "collection_name": self.collection_name,
            "total_items": self._collection.count(),
            "db_path": str(self.db_path),
        }
