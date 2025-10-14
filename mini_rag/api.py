from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from mini_rag.crawler import crawl_site
from mini_rag.indexer import build_index, load_index, search_index

app = FastAPI(title="Mini RAG Service")

index = None
corpus = None
embedder = None


class CrawlRequest(BaseModel):
    start_url: str
    max_pages: int = 10
    max_depth: int = 2
    crawl_delay_ms: int = 500


class IndexRequest(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"


class QueryRequest(BaseModel):   
    question: str
    top_k: int = 5

@app.post("/crawl")
async def crawl(req: CrawlRequest):
    pages, skipped, urls = crawl_site(req.start_url, req.max_pages, req.max_depth, req.crawl_delay_ms)
    return {"page_count": len(pages), "skipped_count": skipped, "urls": urls}

@app.post("/index")
async def index(req: IndexRequest):
    chunks, index_path = build_index(req.chunk_size, req.chunk_overlap, req.embedding_model)
    return {"chunks": chunks, "index_path": index_path}

@app.post("/query")
async def query(req: QueryRequest):
    global index, corpus, embedder

    if index is None or corpus is None or embedder is None:
        from mini_rag.indexer import load_index
        index, corpus, embedder = load_index()

    if index is None or corpus is None:
        raise HTTPException(status_code=400, detail="Please run /index first.")

    results = search_index(req.question, index, corpus, embedder, req.top_k)

    if not results:
        return {"answer": "Not enough information found in crawled content.", "sources": []}

    context = " ".join([r["text"] for r in results])
    generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = (
        "Answer only using this context. "
        "If the context does not contain enough info, say 'not enough information'.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}"
    )
    answer = generator(prompt, max_new_tokens=120)[0]["generated_text"]

    sources = [{"url": r["url"], "snippet": r["text"][:200]} for r in results]
    return {"answer": answer, "sources": sources}