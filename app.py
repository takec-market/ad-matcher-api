from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from sentence_transformers import util
import numpy as np

app = FastAPI()

# ✅ Colab で立てたベクトル変換APIのURL
COLAB_EMBED_URL = "https://c64d-35-233-132-179.ngrok-free.app/embed"

# 広告データ（事前にベクトル化しておいてもOK）
ads = [
    {
        "title": "朝ランイベント in 代々木公園",
        "message": "自然と運動が好きな朝活派向けのランニングイベントです",
        "area": "東京都渋谷区"
    },
    {
        "title": "ネコカフェ渋谷店",
        "message": "猫と静かに過ごしたい人のための癒やし空間です",
        "area": "東京都渋谷区"
    },
    {
        "title": "渋谷バー巡りガイド",
        "message": "ナイトライフを楽しみたい人向けのバー特集",
        "area": "東京都渋谷区"
    }
]

# ユーザー入力の構造定義
class MatchRequest(BaseModel):
    location: str
    intent: str
    profile: str
    top_k: int = 3

# 🔧 テキスト → ベクトル変換（ColabのAPIを使う）
def get_embedding(text: str):
    response = requests.post(COLAB_EMBED_URL, json={"text": text})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch embedding")
    return np.array(response.json()["embedding"])

@app.post("/match")
def match_ads(req: MatchRequest):
    user_text = f"{req.intent}。{req.profile}"
    user_vec = get_embedding(user_text)

    results = []
    for ad in ads:
        ad_vec = get_embedding(ad["message"])  # 毎回変換でもOK（少ない数なら）
        similarity = util.cos_sim(user_vec, ad_vec).item()
        area_bonus = 0.15 if req.location in ad["area"] else 0.0
        total_score = similarity + area_bonus

        results.append({
            "title": ad["title"],
            "message": ad["message"],
            "area": ad["area"],
            "semantic_score": round(similarity, 3),
            "area_bonus": area_bonus,
            "total_score": round(total_score, 3)
        })

    sorted_results = sorted(results, key=lambda x: x["total_score"], reverse=True)[:req.top_k]
    return {"matches": sorted_results}