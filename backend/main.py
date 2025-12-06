# backend/main.py
import os
import uuid
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .chunking import extract_text_by_page, chunk_pages_to_chunks
from .gemini_client import embed_texts, call_gemini_chat
from .faiss_store import (
    ensure_dir,
    create_faiss_index,
    add_embeddings_to_index,
    load_index,
)

# make sure vector dir exists
ensure_dir()

app = FastAPI()

# ---- CORS so frontend (index.html) can call this API ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # for dev: allow all; later you can restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    1) Save uploaded PDF
    2) Extract text, chunk (500 with 100 overlap)
    3) Embed chunks with Gemini
    4) Store embeddings in FAISS + save metadata
    """
    contents = await file.read()
    doc_id = str(uuid.uuid4())

    os.makedirs("uploads", exist_ok=True)
    pdf_path = os.path.join("uploads", f"{doc_id}.pdf")
    with open(pdf_path, "wb") as f:
        f.write(contents)

    # 1. Extract pages
    pages = extract_text_by_page(pdf_path)

    # 2. Chunk into 500-token chunks with 100 overlap
    chunks = chunk_pages_to_chunks(pages, chunk_size=500, overlap=100)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    texts = [c["text"] for c in chunks]
    metadata = [
        {
            "doc_id": doc_id,
            "chunk_id": c["chunk_id"],
            "page_num": c["page_num"],
            "text": c["text"],
        }
        for c in chunks
    ]

    # 3. Get embeddings in batches (avoid giant single request)
    batch_size = 16
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_texts(batch)
        embeddings.extend(embs)

    if not embeddings:
        raise HTTPException(status_code=500, detail="No embeddings created")

    # infer embedding dim from first vector (we saw 3072 in your test)
    embedding_dim = len(embeddings[0])

    # 4. Create FAISS index + save metadata
    index = create_faiss_index(embedding_dim)
    add_embeddings_to_index(index, embeddings, metadata, doc_id)

    return {"doc_id": doc_id, "num_chunks": len(chunks)}


@app.post("/ask")
async def ask_doc(payload: dict):
    """
    Request body example:
    {
      "doc_id": "e192532e-e882-4a95-86cf-03560cba9588",
      "question": "What is the main topic of this document?"
    }
    """
    doc_id = payload.get("doc_id")
    question = payload.get("question")

    if not doc_id or not question:
        raise HTTPException(status_code=400, detail="doc_id and question required")

    # Load FAISS index + metadata
    index, meta = load_index(doc_id)
    if index is None:
        raise HTTPException(status_code=404, detail="doc not found")

    # 1) Embed the question
    q_emb = embed_texts([question])[0]

    # 2) Convert to numpy and search in FAISS
    q_emb = embed_texts([question])[0]

    # number of vectors stored in FAISS
    n_vectors = index.ntotal
    if n_vectors == 0:
        raise HTTPException(status_code=500, detail="No vectors in index for this document")

    top_k = min(5, n_vectors)

    q_vec = np.array([q_emb], dtype="float32")
    D, I = index.search(q_vec, top_k)

    hits = []
    for score, idx in zip(D[0], I[0]):
        # FAISS may return -1 / very negative scores for invalid entries → skip them
        if idx < 0:
            continue
        if score < -1e20:  # filter out the -3.4e38 case
            continue

        row = meta.iloc[idx].to_dict()
        row["score"] = float(score)
        hits.append(row)


    # 4) Build context from hits
    context_parts = []
    for h in hits:
        context_parts.append(f"[page {h['page_num']}] {h['text']}")
    context_str = "\n\n".join(context_parts)

    # 5) Build prompt
    system_prompt = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't know' and do not guess. "
        "Always cite the source pages at the end using [page X]."
    )

    full_prompt = f"""{system_prompt}

Context:
{context_str}

Question: {question}

Answer:"""

    # 6) Call Gemini
    answer = call_gemini_chat(full_prompt)

    return {
        "answer": answer,
        "sources": [
            {"page": h["page_num"], "chunk_id": h["chunk_id"], "score": h["score"]}
            for h in hits
        ],
    }
