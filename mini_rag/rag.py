from mini_rag.crawler import load_crawled_pages
from mini_rag.indexer import build_index, load_index, search_index
from transformers import pipeline
import numpy as np

def full_pipeline(url, question, top_k=5):
    pages = load_crawled_pages()

    if not pages:
        return {"answer": "No crawled content found. Please run /crawl first.", "sources": []}

    index, corpus, model = load_index()

    if index is None:
        build_index(800, 120, "sentence-transformers/all-MiniLM-L6-v2")
        index, corpus, model = load_index()

    results = search_index(question, index, corpus, model, top_k)

    threshold = 0.25
    filtered = [r for r in results if r["score"] > threshold]

    if not filtered:
        return {
            "answer": "Not enough information found in crawled content.",
            "sources": [],
        }

    context = " ".join([r["text"] for r in filtered])

    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = (
        "Answer only using this context. "
        "If context does not contain enough information, say 'not enough information'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    answer = generator(prompt, max_new_tokens=120)[0]["generated_text"]

    sources = [{"url": r["url"], "snippet": r["text"][:200]} for r in filtered]

    return {"answer": answer, "sources": sources}