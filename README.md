<div align="center">

<br/>

<pre>
██████╗  █████╗  ██████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║██║  ███╗
██╔══██╗██╔══██║██║   ██║
██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
</pre>

# 📄 RAG PDF Q&A Bot

### *Ask your PDF anything. Get answers with page references.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-Embeddings-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-F55036?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-00B8D9?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

<br/>

> **Upload a PDF → Ask a Question → Get a Precise Answer with Page References**
>
> *Powered by Google Gemini Embeddings · Groq Llama 3.1 · FAISS Vector Search*

<br/>

---

</div>

<br/>

## 🧠 What is This?

A **Retrieval-Augmented Generation (RAG)** application that lets you have an intelligent conversation with any PDF document. Instead of generic AI guessing from the internet, this bot reads *your* document, understands it semantically, and answers from it — with exact page citations.

<br/>

## ✨ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   📤 Upload PDF                                                 │
│        │                                                        │
│        ▼                                                        │
│   📃 Extract Text  ──►  🔪 Chunk into Pieces                   │
│                               │                                 │
│                               ▼                                 │
│                    🔢 Embed via Gemini API                      │
│                               │                                 │
│                               ▼                                 │
│                    🗃️  Store in FAISS Index                     │
│                                                                 │
│   ❓ You Ask a Question                                         │
│        │                                                        │
│        ▼                                                        │
│   🔢 Embed Question  ──►  🔍 Search FAISS for Top Chunks       │
│                               │                                 │
│                               ▼                                 │
│              🤖 Groq Llama 3.1 reads Context + Question        │
│                               │                                 │
│                               ▼                                 │
│              💬 Answer + 📖 Page References returned           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br/>

## 🔬 What is RAG?

**RAG = Retrieval-Augmented Generation**

Rather than asking an LLM what's inside your document (and hoping it guesses correctly), RAG follows a two-step approach:

| Step | Name | What Happens |
|------|------|--------------|
| 1️⃣ | **Retrieval** | PDF is chunked → embedded into vectors → your question is embedded → most similar chunks are retrieved |
| 2️⃣ | **Generation** | Retrieved chunks (context) + your question → sent to LLM → answer generated from *your document only* |

> The model doesn't guess from general internet knowledge. It answers using **your uploaded document**.

<br/>

---

## 🛠️ Tech Stack

### 🔧 Backend

| Tool | Role |
|------|------|
| **Python** | Core programming language |
| **FastAPI** | Web framework — exposes `/upload`, `/ask`, and serves the frontend |
| **Uvicorn** | ASGI server that runs the FastAPI application |
| **PyMuPDF (`fitz`)** | Reads PDF files and extracts text page-by-page |
| **FAISS (`faiss-cpu`)** | Facebook/Meta vector database — stores embeddings and enables fast similarity search |
| **Pandas** | Stores chunk metadata (page number, text) in structured tables |
| **Tiktoken** | Handles token/length management during text chunking |
| **Requests** | Sends HTTP calls to Gemini and Groq APIs |

<br/>

### 🤖 AI / LLMs

| Model | Provider | Purpose |
|-------|----------|---------|
| **Gemini Embedding API** | Google | Converts text chunks and questions into semantic vectors |
| **Llama 3.1 (`llama-3.1-8b-instant`)** | Groq | Reads context + question → generates the final answer |

<br/>

### 🎨 Frontend

| Tool | Role |
|------|------|
| **HTML + CSS + JavaScript** | Plain, no-framework web UI |
| **`frontend/index.html`** | Upload PDF · Ask questions · View chat-style answers with page refs |

<br/>

### 🧰 Dev / Other

| Tool | Role |
|------|------|
| **Git + GitHub** | Version control and repository hosting |
| **ngrok** *(optional)* | Creates a public URL pointing to your local server — share your bot with anyone for free |

<br/>

---

## 🗂️ Project Structure

```
rag-app/
│
├── backend/
│   ├── main.py              # FastAPI app: routes /, /upload, /ask
│   ├── gemini_client.py     # Gemini (embeddings) + Groq (chat) clients
│   ├── chunking.py          # PDF text extraction + chunk splitting
│   └── faiss_store.py       # FAISS index creation, loading & metadata
│
├── frontend/
│   └── index.html           # Web UI — upload PDF, ask questions, view answers
│
├── uploads/                 # Uploaded PDFs (created at runtime)
├── vector_store/            # FAISS index + metadata per document (created at runtime)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

<br/>

---

## 🚀 Running the App

### On Your Laptop (Local)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/rag-app.git
cd rag-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys
export GEMINI_API_KEY="your-gemini-key"
export GROQ_API_KEY="your-groq-key"

# 4. Start the server
uvicorn backend.main:app --reload

# 5. Open your browser
# http://localhost:8000
```

<br/>

### Share Over the Internet (ngrok)

```bash
# After starting the server, in a new terminal:
ngrok http 8000

# ngrok gives you a public URL like:
# https://abc123.ngrok.io  ← share this with anyone
```

<br/>

---

## 🔑 API Keys You Need

| Service | What For | Get It |
|---------|----------|--------|
| **Google Gemini** | Text embeddings | [ai.google.dev](https://ai.google.dev) |
| **Groq** | LLM inference (Llama 3.1) | [console.groq.com](https://console.groq.com) |

<br/>

---

## 💡 Key Design Decisions

- **Why FAISS?** — Fast, local, no server needed. Stores vectors in-memory and persists to disk per document.
- **Why Gemini for Embeddings?** — High-quality semantic vectors that capture meaning beyond keywords.
- **Why Groq for Chat?** — Extremely fast inference for Llama 3.1, providing near-instant answers.
- **Why Plain HTML Frontend?** — Zero build step, no framework complexity. Just open and use.
- **Why PyMuPDF?** — Reliable page-by-page text extraction from PDFs of all kinds.

<br/>

---

<div align="center">

**Built with 🧠 RAG · 🔥 Groq · 🌐 Google Gemini · ⚡ FastAPI**

*Upload. Ask. Discover.*

</div>
