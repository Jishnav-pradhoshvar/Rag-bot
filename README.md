📄 RAG PDF Q&A Bot (Gemini + Groq + FastAPI) : 

Ask questions about "Your own PDF" and get answers with page references.

This project:

- Lets you "upload a PDF"
- Breaks it into small chunks
- Turns each chunk into "vectors" (numbers) using "Google Gemini embeddings"
- Stores them in a "FAISS" vector database
- When you ask a question , it:
  - finds the most relevant chunks
  - sends them (as context) + your question to ""Groq Llama 3.1""
  - shows the answer + which pages it used

You can run it:

- on "Your own laptop" (localhost)
- and optionally share it over the internet using "ngrok" for free
  

Tech Stack (Tools we used and why)

BACKEND :

- Python – main programming language
- FastAPI – web framework for building APIs (`/upload`, `/ask`, and serving the website)
- Uvicorn – runs the FastAPI app (the actual server)
- PyMuPDF (fitz) – reads PDF files and extracts text page-by-page
- FAISS (faiss-cpu) – vector database from Facebook/Meta; stores embeddings and lets us search “similar” chunks fast
- Pandas – stores metadata about chunks (page number, text, etc.) in a table
- Tiktoken – helps with token/length management when chunking text
- Requests – sends HTTP requests to Gemini and Groq APIs

AI / LLMs :

- Google Gemini Embedding API  
  - Used only for embeddings  
  - Converts text (chunks + questions) into vectors (lists of numbers)
- Groq Llama 3.1 (llama-3.1-8b-instant)  
  - Used for " chat / answering questions " 
  - Reads the context (top chunks) + question and generates the final answer

Frontend

- Plain HTML + CSS + JavaScript (`frontend/index.html`)
  - Simple web page:
    - Upload PDF
    - Ask question
    - See chat-style answers + page references

Dev / Other

- Git + GitHub – version control and hosting the repo
- ngrok (optional) – creates a public URL that points to your local server so others can use it
  

What is RAG ? 

RAG = Retrieval-Augmented Generation.

Instead of asking an LLM directly:

  “What’s inside my PDF ?”

We do two steps:

1. Retrieval  
   - Break PDF into pieces  
   - Embed each piece into a vector  
   - When you ask a question , embed the question and search for the most similar pieces  
   - Get those top pieces (“context”)

2.  Generation  
   - Give that "Context + question" to an LLM  
   - LLM generates answer based only on that context

So the model doesn’t “Guess” from general internet ; it answers using "Your uploaded document".


Project Structure

```text 
rag-app/
├─ backend/
│  ├─ main.py            # FastAPI app: routes /, /upload, /ask
│  ├─ gemini_client.py   # Talks to Gemini (embeddings) and Groq (chat)
│  ├─ chunking.py        # Extracts text from PDF and splits into chunks
│  ├─ faiss_store.py     # Creates + loads FAISS index, saves metadata
│  └─ ... (other helpers if any)
│
├─ frontend/
│  └─ index.html         # Simple web UI (upload + ask)
│
├─ uploads/              # PDFs uploaded (created at runtime)
├─ vector_store/         # FAISS index + metadata per document (created at runtime)
├─ requirements.txt      # Python dependencies
└─ README.md             # This file
