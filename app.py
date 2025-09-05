import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, Query, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from typing import Any
import uvicorn
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# ======================
# LOAD ENV VARIABLES
# ======================
load_dotenv()  # Loads API_KEY from .env
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("❌ API_KEY is not set. Please configure it in .env.")

# ======================
# DOWNLOAD DATA FROM HF DATASET
# ======================
print("🔄 Downloading dataset files from Hugging Face...")

ID_MAPPING_FILE = hf_hub_download(
    repo_id="shekharrao150/platrr-dataset",
    filename="id_mapping.pkl",
    repo_type="dataset"
)

FAISS_INDEX_FILE = hf_hub_download(
    repo_id="shekharrao150/platrr-dataset",
    filename="recipes.index",
    repo_type="dataset"
)

# ======================
# LOAD DATA
# ======================
print("🔄 Loading pickle file...")
with open(ID_MAPPING_FILE, "rb") as f:
    id_mapping = pickle.load(f)
print(f"✅ Loaded {len(id_mapping)} recipes.")

print("🔄 Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_FILE)
print("✅ FAISS index loaded!")

# ======================
# FASTAPI APP
# ======================
app = FastAPI(title="Recipe Search API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return api_key

# ======================
# HELPERS
# ======================
def normalize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower().strip()

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    return obj

# ======================
# SEARCH FUNCTIONS
# ======================
def search_exact_matches(query_norm: str, top_k: int):
    results = []
    for idx, entry in enumerate(id_mapping):
        title = entry.get("title", "")
        ingredients = entry.get("ingredients", "")
        if query_norm in normalize_text(title) or query_norm in normalize_text(ingredients):
            results.append({
                "id": idx,
                "title": title,
                "directions": entry.get("directions", ""),
                "ingredients": ingredients
            })
            if len(results) >= top_k:
                break
    return results

def search_semantic(query_vector: np.ndarray, top_k: int):
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx == -1 or idx >= len(id_mapping):
            continue
        entry = id_mapping[idx]
        results.append({
            "id": to_serializable(idx),
            "title": entry.get("title", ""),
            "directions": entry.get("directions", ""),
            "ingredients": entry.get("ingredients", ""),
            "score": to_serializable(score)
        })
    return results

def embed_query(query: str) -> np.ndarray:
    # 🔥 Placeholder: Replace with your real embedding model
    np.random.seed(abs(hash(query)) % (10**6))
    return np.random.rand(1, index.d).astype("float32")

# ======================
# ROUTES
# ======================
@app.get("/")
def root():
    return {"message": "✅ Recipe Search API is running! Visit /docs for usage."}

@app.get("/health")
def health():
    return {"status": "ok", "recipes_loaded": len(id_mapping)}

@app.get("/search")
def search_recipes(
    query: str = Query(..., description="Search query (ingredient or title)"),
    top_k: int = Query(5, description="Number of results"),
    api_key: str = Depends(get_api_key)
):
    query_norm = normalize_text(query)
    exact_matches = search_exact_matches(query_norm, top_k)
    query_vector = embed_query(query)
    semantic_matches = search_semantic(query_vector, top_k)
    return {
        "query": query,
        "exact_matches": exact_matches,
        "semantic_matches": semantic_matches
    }

# ======================
# MAIN
# ======================
if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
