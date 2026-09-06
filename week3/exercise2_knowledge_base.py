import os
import json
import requests

CODEBASE_DIR = "sample_codebase"
OUTPUT_FILE = "knowledge_base.json"
EMBED_MODEL = "nomic-embed-text"

def chunk_file(filepath: str, chunk_size: int = 300) -> list[str]:
    """Splits a file's content into fixed-size text chunks."""
    with open(filepath, "r") as f:
        content = f.read()
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

def get_embedding(text: str) -> list[float]:
    """Gets an embedding vector for a piece of text using Ollama."""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def build_knowledge_base():
    knowledge_base = []
    for filename in os.listdir(CODEBASE_DIR):
        filepath = os.path.join(CODEBASE_DIR, filename)
        chunks = chunk_file(filepath)
        for idx, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            knowledge_base.append({
                "file": filename,
                "chunk_id": idx,
                "text": chunk,
                "embedding": embedding
            })
            print(f"Embedded {filename} chunk {idx}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(knowledge_base, f)
    print(f"\nKnowledge base saved to {OUTPUT_FILE} with {len(knowledge_base)} chunks.")

if __name__ == "__main__":
    build_knowledge_base()
