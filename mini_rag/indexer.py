import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_index(chunk_size=1000, chunk_overlap=120, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Reads crawled pages from artifacts/pages/combined_text.json,
    splits them into chunks, computes embeddings, and builds FAISS index.
    """

    input_path = "artifacts/pages/combined_text.json"
    if not os.path.exists(input_path):
        raise FileNotFoundError("No crawled data found. Run /crawl first.")

    with open(input_path, "r") as f:
        pages = json.load(f)

    corpus = []
    for page in pages:
        text = page["text"]
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i : i + chunk_size]
            corpus.append({"url": page["url"], "text": chunk})

    print(f"[INFO] Created {len(corpus)} text chunks for indexing")

    model = SentenceTransformer(embedding_model)
    embeddings = model.encode([c["text"] for c in corpus], show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    os.makedirs("artifacts/vectors", exist_ok=True)
    faiss.write_index(index, "artifacts/vectors/index.faiss")
    with open("artifacts/vectors/corpus.json", "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"[DONE] Indexed {len(corpus)} chunks → artifacts/vectors/index.faiss")

    return len(corpus), "artifacts/vectors/index.faiss"


def load_index():
    """Load FAISS index, corpus, and model for querying."""
    index = faiss.read_index("artifacts/vectors/index.faiss")
    with open("artifacts/vectors/corpus.json", "r") as f:
        corpus = json.load(f)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, corpus, model


def search_index(query, index, corpus, model, top_k=5):
    """Perform semantic search over FAISS index."""
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), top_k)

    results = []
    for idx in I[0]:
        if idx < len(corpus):
            results.append(corpus[idx])
    return results