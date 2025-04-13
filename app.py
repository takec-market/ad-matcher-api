from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from sentence_transformers import util
import os
import json

app = FastAPI()

# 🔗 Colabで立てた埋め込みAPI（最新ngrok URLに更新！）
COLAB_EMBED_URL = "https://ac33-34-125-139-229.ngrok-free.app"

# 🍽 レシピデータ（タイトル・説明・カテゴリ）
recipes = [
    {
        "title": "鶏むね肉の香草グリル",
        "description": "ヘルシー志向で高タンパクな食事を求める人におすすめのシンプルグリル料理です。",
        "category": "ヘルシー"
    },
    {
        "title": "スパイシー豆カレー",
        "description": "ベジタリアンやスパイス好きな人にぴったりの栄養満点レシピです。",
        "category": "ベジタリアン"
    },
    {
        "title": "たっぷり野菜のミネストローネ",
        "description": "野菜不足を感じている人や体を温めたい日におすすめのスープです。",
        "category": "野菜"
    },
    {
        "title": "10分でできる和風パスタ",
        "description": "忙しいけどしっかり食べたい人に向けた、時短＆満足感のあるレシピです。",
        "category": "時短"
    },
    {
        "title": "濃厚チョコレートブラウニー",
        "description": "甘いものが好きで、自分へのご褒美が欲しい日にぴったりなスイーツです。",
        "category": "スイーツ"
    }
]

# 💾 ベクトルキャッシュファイル
CACHE_FILE = "recipe_vectors.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        recipe_cache = json.load(f)
else:
    recipe_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(recipe_cache, f)

# 🔁 ベクトル取得（Colab + キャッシュ + 安定化）
def get_embedding(text: str, cache_key: str = None):
    if cache_key and cache_key in recipe_cache:
        return np.array(recipe_cache[cache_key])

    try:
        res = requests.post(COLAB_EMBED_URL, json={"text": text}, timeout=10)
        res.raise_for_status()  # ステータスエラー検出
        data = res.json()       # JSONDecodeError対策
        vec = data["embedding"]
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Colab request failed: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Colab returned invalid JSON")
    except KeyError:
        raise HTTPException(status_code=500, detail="Missing 'embedding' in response")

    if cache_key:
        recipe_cache[cache_key] = vec
        save_cache()

    return np.array(vec)

# 📬 ユーザー入力
class MatchRequest(BaseModel):
    intent: str  # 今どんなものが食べたいか
    profile: str  # 食の好み・傾向
    top_k: int = 3  # 何件返すか

@app.post("/match")
def match_recipes(req: MatchRequest):
    user_text = f"{req.intent}。{req.profile}"
    user_vec = get_embedding(user_text)

    results = []
    for recipe in recipes:
        recipe_vec = get_embedding(recipe["description"], cache_key=recipe["title"])
        similarity = util.cos_sim(user_vec, recipe_vec).item()
        results.append({
            "title": recipe["title"],
            "description": recipe["description"],
            "category": recipe["category"],
            "score": round(similarity, 3)
        })

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:req.top_k]
    return {"matches": sorted_results}