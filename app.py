from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# モデル読み込み
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# --- 広告データ（ここでは固定デモ） ---
ads = [
    {"title": "朝ランイベント in 代々木公園", "message": "自然と運動が好きな朝活派向けのランニングイベントです", "area": "東京都渋谷区"},
    {"title": "ネコカフェ渋谷店", "message": "猫と静かに過ごしたい人のための癒やし空間です", "area": "東京都渋谷区"},
    {"title": "渋谷バー巡りガイド", "message": "ナイトライフを楽しみたい人向けのバー特集", "area": "東京都渋谷区"},
    {"title": "キャンプ用品セール", "message": "自然の中で過ごすのが好きな人に向けたアウトドア用品のセールです", "area": "全国"},
    {"title": "読書カフェ梅田店", "message": "静かに本が読めるカフェでリラックスした時間を提供します", "area": "大阪府梅田"},
    {"title": "猫カフェなんば", "message": "猫が好きな人向けの癒しのカフェです", "area": "大阪府なんば"},
    {"title": "クラブイベント新宿", "message": "音楽とお酒が好きな人向けのナイトクラブイベント", "area": "東京都新宿区"},
    {"title": "ヨガ体験イベント", "message": "自然と調和したい人のためのヨガ体験教室です", "area": "東京都中野区"},
    {"title": "ヘルシー弁当デリバリー", "message": "忙しい現代人の健康を考えた宅配弁当", "area": "全国"},
]

# --- リクエストデータ形式定義 ---
class MatchRequest(BaseModel):
    location: str
    intent: str
    profile: str
    top_k: int = 3  # 返す件数（任意指定）

# --- マッチングAPI ---
@app.post("/match")
async def match_ads(request: MatchRequest):
    user_text = f"{request.intent}。{request.profile}"
    user_vec = model.encode(user_text, convert_to_tensor=True)

    results = []
    for ad in ads:
        ad_vec = model.encode(ad["message"], convert_to_tensor=True)
        similarity = util.cos_sim(user_vec, ad_vec).item()
        area_bonus = 0.15 if request.location in ad["area"] else 0.0
        total_score = similarity + area_bonus

        results.append({
            "title": ad["title"],
            "message": ad["message"],
            "area": ad["area"],
            "semantic_score": round(similarity, 3),
            "area_bonus": area_bonus,
            "total_score": round(total_score, 3)
        })

    sorted_results = sorted(results, key=lambda x: x["total_score"], reverse=True)[:request.top_k]
    return {"matches": sorted_results}
