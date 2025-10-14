# 🧠 Mini RAG (Retrieval-Augmented Generation) Service

🧠 What Is RAG (Retrieval-Augmented Generation)?

RAG is a technique that combines information retrieval (search) with text generation (language models).
Instead of making up answers from memory, the system:
	1.	Retrieves factual information from external sources (like web pages or documents).
	2.	Generates answers grounded in those retrieved facts.

So it’s like giving the model its own short-term “research assistant” — it first finds the right snippets, then explains them in natural language.

⸻

⚙️ Your Mini-RAG System — Step by Step

Your project builds a complete end-to-end RAG pipeline, using open-source components, running locally.

⸻

1️⃣ Crawl – Data Collection
	•	Input: A starting URL, e.g.,
https://en.wikipedia.org/wiki/Artificial_intelligence
	•	The crawler visits that page and follows in-domain links (staying within the same website).
	•	It respects:
	•	The website’s robots.txt
	•	A crawl delay so it doesn’t overload the site.
	•	It extracts visible text from each page (ignores scripts, styles, etc.).
	•	Output: A JSON file — artifacts/pages/combined_text.json
containing a list of pages with: { "url": "https://...", "text": "cleaned text of the page" }

2️⃣ Index – Embedding & Vectorization
	•	The text is split into chunks (≈800 characters each with small overlap).
	•	Each chunk is turned into a numerical vector using the model
sentence-transformers/all-MiniLM-L6-v2.
For example:
"Artificial intelligence is the simulation of human intelligence..."
→ [0.124, -0.562, 0.312, ...]
	•	All these vectors are stored in a FAISS vector database, which lets you find semantically similar text quickly.

3️⃣ Query – Question Answering

When a user asks a question (e.g. “What is Artificial Intelligence?”):
	1.	The question is embedded into a vector using the same model.
	2.	The FAISS index finds the top-k closest chunks (most relevant text).
	3.	These chunks are combined into a context block.
	4.	A local text generation model (flan-t5-base) is prompted: 
            Answer using only this context:
            <retrieved text chunks>
            Question: What is Artificial Intelligence?
    5.	The model generates a grounded answer, using only that context.
	6.	The system returns:
            •	The answer
            •	The source URLs
            •	The retrieval + generation timings

User Question
   ↓
  Embedding (MiniLM)
   ↓
Vector Search (FAISS)
   ↓
Top-k Relevant Chunks
   ↓
Prompt to Generator (Flan-T5)
   ↓
Grounded Answer + Source URLs

## 🚀 Features
- Local-only (no API keys required)
- Crawl within same domain politely
- Chunking: 1000 chars, overlap 120
- Embeddings: `all-MiniLM-L6-v2`
- Answer generation: `flan-t5-base`
- Refusal when context is insufficient
- FastAPI endpoints: `/crawl`, `/index`, `/query`

## ⚙️ Run Locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn mini_rag.api:app --reload
```

## 🧩 Example Usage
```bash
curl -X POST http://127.0.0.1:8000/crawl   -H "Content-Type: application/json"   -d '{"start_url": "https://en.wikipedia.org/wiki/Artificial_intelligence", "max_pages": 50, "max_depth": 10}'

curl -X POST http://127.0.0.1:8000/index   -H "Content-Type: application/json"   -d '{"chunk_size":1000,"chunk_overlap":120,"embedding_model":"sentence-transformers/all-MiniLM-L6-v2"}'

curl -X POST http://127.0.0.1:8000/query   -H "Content-Type: application/json"   -d '{"question":"What is Artificial Intelligence?"}'
```# Mini-RAG
# MINI_RAG
# MINI_RAG
# MINI_RAG
