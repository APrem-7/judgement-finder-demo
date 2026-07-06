# KanoonSaathi — Demo

A demo legal-tech pipeline for Indian case law: bulk ingestion of judgments, PII anonymization,
LLM-generated case summaries (via Groq), semantic "judgement finder" search, and side-by-side
comparison of different LLMs on summarization quality.

- **Backend**: FastAPI + SQLAlchemy (SQLite) + FAISS vector store + spaCy anonymizer + Groq LLM client
- **Frontend**: React (Vite) + Tailwind + Recharts, with four views: Dashboard, Ingestion Pipeline,
  Judgement Finder, Model Comparison

## Prerequisites

- Python 3.10+ (tested on 3.12)
- Node.js 18+ and npm
- A free [Groq API key](https://console.groq.com/keys)

## Setup

### Option A — one-shot script (Windows)

```bat
setup.bat
```

This installs backend Python deps, downloads the spaCy `en_core_web_sm` model, creates the
`storage/` and `data/` directories, and runs `npm install` in `frontend/`.

### Option B — manual

```bash
# Backend
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd ..

# Frontend
cd frontend
npm install
cd ..
```

### Configure your Groq API key

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and set `GROQ_API_KEY=<your key>`. This file is gitignored — never commit
real API keys.

## Running the demo

Start both servers (two terminals), or use the provided scripts on Windows:

```bat
start_backend.bat
start_frontend.bat
```

Or manually:

```bash
# Terminal 1 — API on http://localhost:8000
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — UI on http://localhost:5173
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

The repo ships with a pre-seeded `kanoonsathi.db` and `storage/` folder (sample cases, generated
summaries, FAISS index), so you can explore Judgement Finder and Model Comparison immediately
without running ingestion first.

## Using the demo

- **Dashboard** — overview stats: cases ingested, documents generated, model tests run, vector
  store size.
- **Ingestion Pipeline** — upload a CSV of raw judgments (see `data/ak_gopalan_case.csv` for the
  expected format). Each row is anonymized, summarized by an LLM, and embedded into the FAISS
  index.
- **Judgement Finder** — semantic search over ingested/anonymized cases using vector similarity.
- **Model Comparison** — run a case through multiple Groq-hosted models and compare summaries on
  completeness, PII safety, readability, structure, and legal-term coverage.

## Project layout

```
backend/
  main.py          # FastAPI app and routes
  config.py        # env-driven settings
  db/              # SQLAlchemy models + session
  pipeline/        # ingestion, anonymization, document generation
  rag/             # embeddings, FAISS vector store, similarity search
  llm/             # Groq client, prompts, model tester
  data/            # sample CSVs
  storage/         # generated documents + FAISS index (demo instance)
frontend/
  src/App.jsx      # routing/layout
  src/pages/       # Dashboard, IngestionPipeline, JudgementFinder, ModelComparison
  src/api/client.js
data/              # sample CSVs (duplicated at repo root for convenience)
storage/           # generated documents + FAISS index (repo-root instance)
kanoonsathi.db     # seeded SQLite demo database
```

## Notes

- This is a demo/prototype, not production-hardened: no auth, permissive CORS for local dev
  origins only, and secrets belong in `backend/.env` (gitignored).
- Backend defaults to SQLite; `DATABASE_URL` in `.env` can be pointed at Postgres if desired.
