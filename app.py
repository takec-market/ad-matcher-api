import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from sentence_transformers import util

app = FastAPI()
COLAB_EMBED_URL = "https://c64d-35-233-132-179.ngrok-free.app"

ADS_FILE = "ad_vectors.json"

ads = [
    {"title": "キャンプ用品セール", "message": "自然の中で過ごすのが好きな人に向けたアウトドア用品のセールです", "area": "全国"},
    {"title": "ネコカフェ渋谷店", "message": "猫と静かに過ごしたい人のための癒やし空間です", "area": "東京都渋谷区"},
    {"title": "渋谷バー巡りガイド", "message": "ナイトライフを楽しみたい人向けのバー特集", "area": "東京都渋谷区"}
]

# 🔄 キャッシュの読み込み
if os.path.exists(ADS_FILE):
    with open(ADS_FILE, "r") as f:
        ad_cache = json.load(f)
else:
    ad_cache = {}

def save_cache():
    with open(ADS_FILE, "w") as f:
        json.dump(ad_cache, f)

def get_embedding(text: str, cache_key: str = None):
    if cache_key and cache_key in ad_cache:
        return np.array(ad_cache[cache_key])
    
    # Colabに投げる
    res = requests.post(COLAB_EMBED_URL, json={"text": text})
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail="Embedding fetch failed")
    
    vec = res.json()["embedding"]
    if cache_key:
        ad_cache[cache_key] = vec
        save_cache()
    return np.array(vec)

class MatchRequest(BaseModel):
    location: str
    intent: str
    profile: str
    top_k: int = 3

@app.post("/match")
def match_ads(req: MatchRequest):
    user_vec = get_embedding(f"{req.intent}。{req.profile}")

    results = []
    for ad in ads:
        ad_vec = get_embedding(ad["message"], cache_key=ad["title"])
        similarity = util.cos_sim(user_vec, ad_vec).item()
        area_bonus = 0.15 if req.location in ad["area"] else 0.0
        total_score = similarity + area_bonus

        results.append({
            "title": ad["title"],
            "score": round(total_score, 3),
            "semantic_score": round(similarity, 3),
            "area_bonus": area_bonus
        })

    return {"matches": sorted(results, key=lambda x: x["score"], reverse=True)[:req.top_k]}
