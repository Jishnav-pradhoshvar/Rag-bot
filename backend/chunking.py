# backend/chunking.py
import fitz  # PyMuPDF
import tiktoken

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        pages.append({"page_num": i+1, "text": txt})
    return pages

def get_tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def text_to_tokens(text, tokenizer):
    if tokenizer:
        return tokenizer.encode(text)
    else:
        return text.split()

def tokens_to_text(tokens, tokenizer):
    if tokenizer:
        return tokenizer.decode(tokens)
    else:
        return " ".join(tokens)

def chunk_pages_to_chunks(pages, chunk_size=500, overlap=100):
    tokenizer = get_tokenizer()
    chunks = []
    for p in pages:
        page_num = p["page_num"]
        text = p["text"]
        token_ids = text_to_tokens(text, tokenizer)
        if not token_ids:
            continue
        step = chunk_size - overlap
        start = 0
        chunk_id = 0
        while start < len(token_ids):
            end = start + chunk_size
            tok_chunk = token_ids[start:end]
            chunk_text = tokens_to_text(tok_chunk, tokenizer)
            chunks.append({
                "page_num": page_num,
                "chunk_id": f"{page_num}_{chunk_id}",
                "start_token": start,
                "end_token": min(end, len(token_ids)),
                "text": chunk_text
            })
            chunk_id += 1
            start += step
    return chunks
