"""
app.py - Python ML microservice for: 
- embeddings (sentence-transformers) 
- FAISS local retrieval (or Pinecone integration) 
- ASR using Whisper or Vosk 
- Translation (HuggingFace Marian / mBART) 
- TTS (Coqui / gTTS) 
""" 
import os 
import json 
import tempfile 
from typing import List 
from fastapi import FastAPI, UploadFile, File 
from pydantic import BaseModel 
from sentence_transformers import SentenceTransformer 
import faiss 
import numpy as np 
from datetime import datetime 
from pathlib import Path 

app = FastAPI(title="AgriNova ML Service") 

ROOT = Path(__file__).resolve().parent 
DATA_DIR = ROOT / "data" 
ADVISORIES_FILE = DATA_DIR / "advisories.json" 

# Load advisories 
try:
    with open(ADVISORIES_FILE, "r", encoding="utf-8") as f: 
        advisories = json.load(f) 
except FileNotFoundError:
    print(f"ERROR: Advisory file not found at {ADVISORIES_FILE}. Retrieval will fail.")
    advisories = []


# sentence-transformers model (change to multilingual/indic as needed) 
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 

# Prepare documents for FAISS if advisories were loaded
if advisories:
    DOCS = [a["text"] for a in advisories] 
    DOC_IDS = [a["id"] for a in advisories] 
    print("Encoding documents...")
    EMBS = EMBED_MODEL.encode(DOCS, convert_to_numpy=True) 

    # FAISS index 
    D = EMBS.shape[1] 
    INDEX = faiss.IndexFlatL2(D) 
    INDEX.add(EMBS)
    print(f"FAISS Index initialized with {len(advisories)} documents.")
else:
    # Create dummy FAISS components if no data was loaded
    INDEX = type('DummyIndex', (object,), {'is_trained': False, 'search': lambda q, k: (np.array([[0]]), np.array([[0]]))})()
    DOCS = []
    DOC_IDS = []
    EMBS = np.array([])
    D = 0
    print("WARNING: FAISS Index not initialized due to missing data.")


# Translation models (HuggingFace Marian or mBART) 
from transformers import pipeline 

# language detection & translation pipeline (use appropriate model for Malayalam <-> English) 
# Example uses Helsinki-NLP Marian models; for Malayalam you might use "Helsinki-NLP/opus-mt-en-mul" or mBART 
# For demo, we use a no-op translator if model not available. 
try: 
    translator_en_to_ml = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul") 
    translator_ml_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en") 
    print("Translation pipelines initialized.")
except Exception as e: 
    translator_en_to_ml = None 
    translator_ml_to_en = None 
    print("Translation pipelines not available (will skip).", e) 

# Optional ASR: Whisper or Vosk (demo uses whisper if installed) 
try: 
    import whisper 
    WHISPER_MODEL = whisper.load_model("small") 
    USE_WHISPER = True 
    print("Whisper ASR model loaded.")
except Exception as e: 
    print("Whisper not available:", e) 
    USE_WHISPER = False 

# TTS: use gTTS as fallback for demo 
from gtts import gTTS 

# Pydantic class 
class RetrieveRequest(BaseModel): 
    text: str 
    top_k: int = 4 
    lang: str = "ml" 

@app.post("/retrieve") 
def retrieve(req: RetrieveRequest): 
    text = req.text 
    top_k = req.top_k 

    if not advisories:
        return {"results": []}

    # embed query 
    q_emb = EMBED_MODEL.encode([text], convert_to_numpy=True) 
    D, I = INDEX.search(q_emb, top_k) 
    
    results = [] 
    for idx, score in zip(I[0], D[0]): 
        if 0 <= idx < len(advisories): 
            adv = advisories[idx] 
            results.append({ 
                "id": adv["id"], 
                "title": adv.get("title", ""), 
                "excerpt": adv["text"][:1000], 
                "score": float(score) 
            }) 
    return {"results": results} 

@app.post("/embed") 
def embed_text(payload: dict): 
    text = payload.get("text", "") 
    v = EMBED_MODEL.encode([text], convert_to_numpy=True)[0].tolist() 
    return {"vector": v} 

@app.post("/transcribe") 
async def transcribe(file: UploadFile = File(...), lang: str = "ml"): 
    # Save to temp file 
    contents = await file.read() 
    # Use the correct suffix based on common audio types for whisper, e.g., mp3 or wav
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav") 
    tmp.write(contents) 
    tmp.flush() 
    tmp_path = tmp.name 
    
    transcribed_text = ""
    if USE_WHISPER: 
        try:
            # Whisper handles the audio file conversion internally
            result = WHISPER_MODEL.transcribe(tmp_path, language=None) # language autodetect 
            transcribed_text = result.get("text", "") 
        except Exception as e:
            print(f"Whisper transcription failed: {e}")
            transcribed_text = f"Transcription failed: {str(e)}"
    else: 
        transcribed_text = "ASR model not available."
    
    # Clean up temp file
    os.unlink(tmp_path)

    return {"text": transcribed_text} 

@app.post("/translate") 
def translate(payload: dict): 
    text = payload.get("text", "") 
    src = payload.get("src", "auto") 
    tgt = payload.get("tgt", "en") 

    # For demo: if translator pipelines available, use; else return input 
    if src.startswith("en") and tgt.startswith("ml") and translator_en_to_ml: 
        out = translator_en_to_ml(text, max_length=400) 
        return {"translation": out[0]['translation_text']} 
    
    if src.startswith("ml") and tgt.startswith("en") and translator_ml_to_en: 
        out = translator_ml_to_en(text, max_length=400) 
        return {"translation": out[0]['translation_text']} 
    
    return {"translation": text} 

@app.post("/tts") 
def tts(payload: dict): 
    text = payload.get("text", "") 
    lang = payload.get("lang", "ml") # gTTS supports 'ml' for Malayalam; if not available, it will fallback to en 

    if not text:
        return {"error": "Text for TTS is missing."}
        
    try: 
        tts_lang = 'ml' if lang.lower().startswith('ml') else 'en' 
        tts = gTTS(text=text, lang=tts_lang) 
        fname = f"tts_{int(datetime.utcnow().timestamp())}.mp3" 
        out_path = str(tempfile.gettempdir() + "/" + fname) 
        tts.save(out_path) 
        return {"filename": fname, "path": out_path} 
    except Exception as e: 
        return {"error": str(e)} 

@app.get("/health") 
def health(): 
    return {"status": "ok", "faiss_ready": INDEX.is_trained if hasattr(INDEX, 'is_trained') else True, "whisper_ready": USE_WHISPER}