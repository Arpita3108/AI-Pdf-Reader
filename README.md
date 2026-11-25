<<<<<<< HEAD
# AI PDF Research Assistant (RAG)

Upload one or many PDFs, ask questions in plain English, and get answers grounded **only** in your documents, with the **document name and page number** for every source.

```
PDF ─► PyMuPDF text ─► clean ─► chunk (size+overlap) ─► Sentence-Transformer embeddings ─► FAISS
                                                                                            │
Question ─► embed ─► top-K similar chunks (cosine ≥ MIN_SIMILARITY) ─► LLM (context only) ─► answer + sources
                          └─ nothing relevant? ─► "I couldn't find that" (LLM is never called)
```

## Run it in VS Code (step by step)

1. **Unzip** the project and open the folder: *File → Open Folder → pdf-research-assistant*.
2. **Install** Python 3.10+ and the VS Code *Python* extension.
3. Open the terminal (**Ctrl+`**) and create a virtual environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Select the interpreter: **Ctrl+Shift+P → Python: Select Interpreter → .venv**.
5. Install dependencies (first time takes a few minutes; torch is large):
   `pip install -r requirements.txt`
6. Create your config: copy `.env.example` to `.env` (Windows: `copy .env.example .env`).
   - **Low-spec laptop / quick try:** set `LLM_PROVIDER=extractive` (no model download, instant).
   - **Best free quality:** install [Ollama](https://ollama.com), run `ollama pull llama3.2`, set `LLM_PROVIDER=ollama`.
   - **Fully inside Python:** keep `LLM_PROVIDER=huggingface` (downloads ~3 GB on first question).
7. **Terminal 1, backend:** `uvicorn app.main:app --reload`
   (API docs at http://localhost:8000/docs)
8. **Terminal 2, frontend** (click the **+** in the terminal panel, activate `.venv` again):
   `streamlit run frontend/streamlit_app.py` → opens http://localhost:8501
9. In the sidebar upload PDFs → **Process uploads** → ask questions in the chat box.

Optional: press **F5** in VS Code and pick *Backend (FastAPI)* or *Frontend (Streamlit)* (see `.vscode/launch.json`).

Run the offline tests (no downloads needed): `pip install pytest` then `pytest -q`

## Project layout

| File | Responsibility |
|---|---|
| `app/config.py` | Settings from `.env` |
| `app/pdf_processor.py` | Validate PDF, extract text per page, clean text, read metadata |
| `app/chunker.py` | Sentence-aware chunking, configurable size/overlap, page metadata on each chunk |
| `app/embeddings.py` | Sentence-Transformer embeddings (normalised) |
| `app/vector_store.py` | FAISS index, similarity search, delete, save/load to `data/` |
| `app/prompts.py` | Grounding rules + citation format |
| `app/llm.py` | Providers: huggingface, ollama, openai-compatible, extractive |
| `app/rag.py` | Ingest + ask pipeline |
| `app/main.py` | FastAPI endpoints |
| `frontend/streamlit_app.py` | Chat UI |
| `tests/test_pipeline.py` | Offline tests |

## How embeddings capture meaning

An embedding model turns a piece of text into a list of numbers (384 for `all-MiniLM-L6-v2`). It is trained so that texts with similar *meaning* land close together in that space, even with different words: "heart attack" lands near "myocardial infarction". Vectors are normalised to length 1, so the dot product of two vectors equals their **cosine similarity**. FAISS finds the chunk vectors with the highest similarity to the question vector, which is why retrieval works on meaning, not just keyword matches.

## How hallucination is limited

1. The LLM only sees retrieved passages and is instructed to answer from them alone, cite `[n]`, and say "not found" otherwise.
2. If no chunk reaches `MIN_SIMILARITY`, the LLM is skipped entirely and a "not found" reply is returned.
3. Sources (document, page, similarity, snippet) are attached by the backend from retrieval metadata, so they are never invented by the model. Sources the model actually cited are marked ✅.
4. Temperature is 0 / greedy decoding.

Small local models can still occasionally misread a passage. Use the source snippets to verify, or use a larger model.

## Tuning

| Setting | Effect |
|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Smaller chunks = more precise retrieval; more overlap = fewer facts cut at boundaries |
| `TOP_K` | More passages = more context for the LLM, but more noise |
| `MIN_SIMILARITY` | Raise (0.35+) to be stricter about "not found"; lower if good answers get rejected |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` is a stronger free alternative; multilingual: `paraphrase-multilingual-MiniLM-L12-v2`. Delete `data/` after changing it |

## Costs

Everything defaults to **free and local**. Paid only if you choose `LLM_PROVIDER=openai`: `gpt-4o-mini` costs a few cents per 100 questions (check current pricing). Free API alternatives with the same setting: Groq or Together free tiers (set `OPENAI_BASE_URL` and `OPENAI_MODEL`).

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | multipart `files` (one or many PDFs) |
| GET | `/documents` | list indexed documents |
| DELETE | `/documents/{name}` | remove a document |
| POST | `/ask` | `{"question": "...", "top_k": 5, "documents": ["a.pdf"]}` |
| GET | `/health` | status |

## Known limitations / next steps

- Scanned PDFs (images) have no text layer and are rejected with a clear message; add OCR (Tesseract / `pymupdf` OCR) to support them.
- Tables and multi-column layouts may extract imperfectly.
- Ideas: cross-encoder re-ranking, hybrid BM25 + vector search, streaming answers, user authentication, Docker.

## Push to GitHub

```
git init
git add .
git commit -m "AI PDF Research Assistant"
git branch -M main
git remote add origin https://github.com/<you>/pdf-research-assistant.git
git push -u origin main
```
`.env` and `data/` are git-ignored.
=======
# AI-Pdf-Reader
ppt of AI pdf Reader
>>>>>>> 4e9061b9ab8dc4230ea78bed6ea2b6ff33d698d2
