import os
import requests

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
EMBED_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "models/gemini-2.0-flash"

def embed_texts(text_list):
    """
    text_list: list[str]
    returns: list[list[float]]  (one embedding per input)
    Uses gemini-embedding-001 via embedContent (one HTTP call per text).
    """
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    url = f"{BASE_URL}/{EMBED_MODEL}:embedContent?key={GEMINI_KEY}"
    headers = {"Content-Type": "application/json"}

    embeddings = []
    for text in text_list:
        payload = {
            "model": EMBED_MODEL,
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        resp = requests.post(url, json=payload, headers=headers)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Print full response for debugging if something goes wrong
            print("Embed error response:", resp.text)
            raise e

        data = resp.json()
        # REST embedContent response (single) → { "embedding": { "values": [...] } }
        emb = data["embedding"]["values"]
        embeddings.append(emb)

    return embeddings


def call_gemini_chat(prompt, max_tokens=512, temperature=0.0):
    """
    Text chat using gemini-1.5-flash.
    """
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment")

    url = f"{BASE_URL}/{CHAT_MODEL}:generateContent?key={GEMINI_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    }
    resp = requests.post(url, json=payload, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("Chat error response:", resp.text)
        raise e
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]
