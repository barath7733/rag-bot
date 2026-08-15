# General-Purpose AI Assistant + RAG Document Intelligence

A production-ready chatbot that works in **three chat modes, plus image generation**:

- **General AI Mode** — answers everyday questions directly via [Groq](https://groq.com)'s LLM API.
- **RAG Document Mode** — answers questions grounded in your uploaded PDF documents, using vector search over [Pinecone](https://www.pinecone.io) and an **open-source, local** embedding model (via `sentence-transformers`).
- **Web Search Mode** — answers questions about current events or anything past the LLM's training cutoff, grounded in fresh results from [Tavily](https://tavily.com), with clickable source links.
- **Image Generation** — generate an image from a text prompt (via the free [Pollinations.ai](https://pollinations.ai) API, no key required) from the sidebar.

The backend automatically decides which chat mode to use for a given question ("Auto" mode), or you can force **General AI**, **Document**, or **Web Search** mode from the UI.

**Zero OpenAI dependency.** Groq is the only LLM provider; embeddings are generated locally with an open-source model — no OpenAI SDK, keys, or models are used anywhere in this project.

---

## 1. Project Overview

| | |
|---|---|
| **LLM provider** | Groq (configurable model) |
| **Vector database** | Pinecone (serverless) |
| **Embeddings** | `sentence-transformers` (open-source, local, configurable) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | HTML / CSS / vanilla JavaScript (ChatGPT/Gemini-style UI) |
| **Document format** | PDF (via `pypdf`) |

## 2. General AI Mode

When a question doesn't need document context, the backend sends it (plus recent conversation history) directly to the configured Groq model and streams the answer back — no retrieval step involved.

## 3. RAG Document Mode

1. You upload a PDF.
2. Text is extracted per page with `pypdf`.
3. Text is cleaned and split into overlapping chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP`).
4. Each chunk is embedded locally with the configured `sentence-transformers` model.
5. Embeddings + metadata (document name, chunk id, page number) are upserted into Pinecone.
6. When you ask a question, the question is embedded the same way, and Pinecone returns the `TOP_K` most similar chunks above `SIMILARITY_THRESHOLD`.
7. Retrieved chunks are passed to Groq as context; Groq is instructed to answer **primarily** from that context and to say clearly when the answer isn't in the documents.
8. The answer, along with source document/page/score metadata, is returned to the frontend.

## 3.5 Web Search Mode

When you select **Web Search** (or Auto detects a current-events question), the backend queries the Tavily search API, passes the top results to Groq as context (same grounding approach as RAG), and returns an answer with clickable source links. Requires `TAVILY_API_KEY`.

## 3.6 Image Generation

The sidebar's **Generate Image** box sends your prompt to `/api/image/generate`, which builds a request to Pollinations.ai's free image API and returns a direct image URL rendered inline in the chat. No API key is required for this by default; see section 7.7 to swap in a paid provider (Stability AI, Together AI, etc.) for higher quality or commercial licensing.

## 4. Architecture

```
User
  → Frontend (HTML/CSS/JS)
  → FastAPI Backend
      → Mode Detection (Auto) or user-selected mode
      → General AI   → Groq LLM → Answer
      → RAG Pipeline  → Pinecone (similarity search) → Retrieved Context → Groq LLM → Answer
      → Web Search    → Tavily API → Retrieved Results → Groq LLM → Answer + Source Links
      → Image Gen     → Pollinations.ai → Image URL
```

## 5. Technology Stack

- Python 3.11+
- FastAPI, Uvicorn
- Groq Python SDK
- Pinecone Python SDK (serverless index)
- `sentence-transformers` for embeddings
- `pypdf` for PDF text extraction
- `python-dotenv`, `pydantic` / `pydantic-settings`
- HTML, CSS, vanilla JavaScript frontend

## 6. Folder Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI app & routes
│   ├── config.py            # Environment-based settings
│   ├── rag.py                # Orchestration: ingest, retrieve, route, answer
│   ├── embeddings.py         # sentence-transformers wrapper
│   ├── pinecone_db.py        # Pinecone index management, upsert, query, delete
│   ├── groq_client.py        # Groq chat completion calls + prompts
│   ├── pdf_processor.py      # PDF text extraction
│   ├── chunking.py           # Cleaning & chunking logic
│   ├── web_search.py         # Tavily web search integration
│   ├── image_gen.py          # Pollinations.ai image generation
│   └── models.py             # Pydantic request/response models
├── data/
│   ├── documents/            # Local JSON registry of indexed documents
│   └── pdfs/                 # (reserved for any on-disk PDF storage)
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 7. Setup

### 7.1 Python & virtual environment

```bash
cd rag-chatbot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 7.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run will download the `sentence-transformers` model (default `all-MiniLM-L6-v2`, ~90 MB) and cache it locally.

### 7.3 Groq API setup

1. Create an account at https://console.groq.com and generate an API key.
2. Set `GROQ_API_KEY` in your `.env`.
3. Set `GROQ_MODEL` to any current Groq-hosted chat model (e.g. `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`). Check https://console.groq.com/docs/models for the current list.

### 7.4 Pinecone setup

1. Create a free account at https://app.pinecone.io and generate an API key.
2. Set `PINECONE_API_KEY` in your `.env`.
3. Choose an index name via `PINECONE_INDEX_NAME` (created automatically on first run if it doesn't exist).
4. Set `PINECONE_CLOUD` / `PINECONE_REGION` to match a region available on your Pinecone plan (defaults: `aws` / `us-east-1`, which is available on the free tier).

You do **not** need to manually create the index or set its dimension — the app determines the embedding model's true output dimension at runtime and creates (or validates) the Pinecone index to match it automatically.

### 7.5 Embedding model setup

Set `EMBEDDING_MODEL` in `.env` to any `sentence-transformers`-compatible model, e.g.:

- `sentence-transformers/all-MiniLM-L6-v2` (384 dims, fast, default)
- `sentence-transformers/all-mpnet-base-v2` (768 dims, higher quality, slower)

If you change the embedding model **after** documents have already been indexed, either delete the Pinecone index or use a new `PINECONE_INDEX_NAME`, since a new model changes the vector dimension.

### 7.6 Web search setup (Tavily)

1. Create a free account at https://tavily.com and generate an API key (free tier includes a monthly quota, no credit card required).
2. Set `TAVILY_API_KEY` in your `.env`.
3. If it's left blank, the app still runs fine — Web Search mode will just return a clear "not configured" message instead of crashing.

### 7.7 Image generation setup

Works out of the box with **no API key** — it calls the free Pollinations.ai image API. Optionally set `IMAGE_GEN_MODEL` (`flux` for quality, `turbo` for speed).

To swap in a paid provider later (e.g. Stability AI or Together AI) for higher quality / commercial licensing, replace the implementation in `app/image_gen.py` with a call to that provider's API — the rest of the app (the `/api/image/generate` endpoint and frontend) doesn't need to change.

### 7.8 Environment variables

```bash
cp .env.example .env
# then edit .env and fill in GROQ_API_KEY and PINECONE_API_KEY
```

Full list of variables (see `.env.example` for defaults):

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (required) |
| `GROQ_MODEL` | Groq chat model name |
| `PINECONE_API_KEY` | Pinecone API key (required) |
| `PINECONE_INDEX_NAME` | Pinecone index name |
| `PINECONE_CLOUD` / `PINECONE_REGION` | Serverless index location |
| `EMBEDDING_MODEL` | `sentence-transformers` model name |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunking parameters (characters) |
| `TOP_K` | Number of chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | Minimum similarity score to accept a retrieved chunk |
| `TAVILY_API_KEY` | Tavily API key (required for Web Search mode) |
| `WEB_SEARCH_MAX_RESULTS` | Number of web results retrieved per query |
| `IMAGE_GEN_MODEL` | Pollinations.ai model (`flux` or `turbo`) |
| `MAX_UPLOAD_SIZE_MB` | Max PDF upload size |
| `ALLOWED_ORIGINS` | CORS origins, comma-separated or `*` |
| `LOG_LEVEL` | Python logging level |

## 8. Local Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

## 9. Uploading Documents

1. Click **Choose PDF** in the sidebar.
2. Wait for the "indexed successfully" confirmation and chunk count.
3. The document appears in the **Documents** list, where you can delete it at any time (this removes its vectors from Pinecone too).

Re-uploading a PDF with the same name and size is detected as a duplicate and rejected unless you explicitly replace it.

## 10. Using General AI Mode

Select **General AI** in the mode toggle, or leave it on **Auto** and ask a question unrelated to your documents (e.g. "Explain quantum entanglement"). The question goes straight to Groq.

## 11. Using Document/RAG Mode

Select **Document** in the mode toggle (or leave it on **Auto** — the backend will detect document-related questions once you've uploaded at least one file), then ask a question about the content of an uploaded PDF. The answer will include a **Sources** section showing which document/page/score backed the answer. If nothing relevant is found, you'll get a clear "not found in the uploaded documents" message instead of a hallucinated answer.

## 11.5 Using Web Search Mode

Select **Web Search** in the mode toggle, then ask about anything current (news, prices, recent releases). The answer will include clickable **Sources** with the page title and a snippet from each result.

## 11.6 Generating Images

Type a description into the **Generate Image** box in the sidebar and click **🎨 Generate Image**. The image appears inline in the chat once generated.

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `GROQ_API_KEY is not configured` | Set it in `.env` and restart the server. |
| `PINECONE_API_KEY is not configured` | Set it in `.env` and restart the server. |
| Web Search mode says "not configured" | Set `TAVILY_API_KEY` in `.env` and restart the server. |
| Image generation fails / times out | Pollinations.ai is a free public service and can be briefly overloaded — retry, or rephrase the prompt if it's being rejected. |
| Pinecone index dimension mismatch error | You changed `EMBEDDING_MODEL` after the index was created. Use a new `PINECONE_INDEX_NAME` or delete the old index in the Pinecone console. |
| "No extractable text was found in this PDF" | The PDF is likely scanned/image-only; OCR is not included in this project. |
| Slow first request | The embedding model downloads and loads into memory on first use — subsequent requests are fast. |
| CORS errors in the browser console | Set `ALLOWED_ORIGINS` to include your frontend's origin. |

Check **`GET /api/health`** at any time for a live status of Groq/Pinecone configuration and the embedding model's actual vector dimension.

## 13. Security Notes

- API keys are read only from environment variables/`.env` and are never sent to or exposed in the frontend.
- Uploaded files are validated by content type and size before processing.
- Corrupted, encrypted, and empty PDFs are handled gracefully with clear error messages.
- Internal errors return a generic message to the client; full details are only logged server-side (never logging credentials).
- Document IDs are randomly generated (not derived from user input) to avoid path traversal or collision issues.

## 14. Deployment

The app is a standard ASGI FastAPI app and deploys the same way anywhere that supports Python:

### Production start command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2
```

### Render / Railway

1. Push this repo to GitHub.
2. Create a new **Web Service**, connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add the environment variables listed above in the service's dashboard (never commit `.env`).

### Hugging Face Spaces (Docker SDK)

Add a minimal `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

Set the same environment variables as **Secrets** in the Space settings.

### AWS (e.g. Elastic Beanstalk / ECS / EC2)

Run the same start command behind your platform's process manager, and set the environment variables via the platform's secret/parameter store rather than a checked-in `.env` file.

---

**Required environment variables at deploy time:** `GROQ_API_KEY`, `GROQ_MODEL`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `EMBEDDING_MODEL` (plus the tuning variables if you want non-default values). `TAVILY_API_KEY` is optional (only needed for Web Search mode) — image generation needs no key at all.
