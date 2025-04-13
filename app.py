from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import json

app = FastAPI()

# 🎡 モデル読み込み（Cloud Run内で一度だけ)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 📂 レシピデータ
recipes = [
    {
        "title": "鶏むね肉の香草グリル",
        "description": "高タンパクで脂肪の少ない鶏むね肉を香草で風味豊かに仕上げた、ダイエット中の方に最適な一品です。",
        "category": "ヘルシー"
    },
    {
        "title": "スパイシー豆カレー",
        "description": "豆とスパイスをふんだんに使った、ベジタリアン向けの栄養たっぷりカレーです。",
        "category": "ベジタリアン"
    },
    {
        "title": "ミネストローネスープ",
        "description": "たっぷりの野菜を煮込んだ体温まるスープ。野菜不足の方や風邪気味のときにぴったりです。",
        "category": "スープ"
    },
    {
        "title": "時短ナポリタン",
        "description": "10分で作れる昔ながらのナポリタン。忙しい日のお昼ごはんに最適です。",
        "category": "時短"
    },
    {
        "title": "チョコレートブラウニー",
        "description": "濃厚なチョコレートを使用したしっとり系ブラウニー。甘党の方におすすめです。",
        "category": "スイーツ"
    },
    {
        "title": "豆腐ハンバーグ",
        "description": "ヘルシー志向な方にぴったりの豆腐入りふわふわハンバーグです。",
        "category": "ヘルシー"
    },
    {
        "title": "玄米と野菜のビビンバ",
        "description": "玄米を使った食物繊維豊富なビビンバ。腹持ちも良く健康的です。",
        "category": "野菜"
    },
    {
        "title": "鯖の味噌煮",
        "description": "EPA・DHAが豊富な鯖を使った味噌煮で、健康を気にする方におすすめ。",
        "category": "和食"
    },
    {
        "title": "ガパオライス",
        "description": "スパイシーでエスニックな味わいが楽しめる、バジル炒めご飯です。",
        "category": "エスニック"
    },
    {
        "title": "白菜と豚肉のミルフィーユ鍋",
        "description": "冬にぴったりの体温まる鍋。白菜と豚肉の旨味がぎっしり。",
        "category": "鍋"
    },
    {
        "title": "シーザーサラダ",
        "description": "ロメインレタスとチーズのシンプルなサラダ。サイドにぴったり。",
        "category": "サラダ"
    },
    {
        "title": "しらすと大葉の和風パスタ",
        "description": "さっぱりとした風味の和風パスタで、軽い食事におすすめです。",
        "category": "パスタ"
    },
    {
        "title": "納豆キムチご飯",
        "description": "腸内環境を整える発酵食品コンボ。スタミナをつけたい朝にどうぞ。",
        "category": "発酵食品"
    },
    {
        "title": "トマトとバジルのカプレーゼ",
        "description": "イタリアンな前菜に最適な、フレッシュモッツァレラとトマトの組み合わせ。",
        "category": "前菜"
    },
    {
        "title": "バターチキンカレー",
        "description": "まろやかなバターのコクとトマトの酸味が絶妙なインド風カレー。",
        "category": "カレー"
    },
    {
        "title": "鮭のホイル焼き",
        "description": "バターときのこ、野菜で蒸し焼きにした香り豊かなホイル焼きです。",
        "category": "魚料理"
    },
    {
        "title": "焼き野菜のマリネ",
        "description": "冷やしても美味しい野菜の常備菜。彩りも豊かでお弁当にも◎",
        "category": "常備菜"
    },
    {
        "title": "ハヤシライス",
        "description": "牛肉と玉ねぎをじっくり煮込んだ定番洋食メニュー。甘辛で子どもにも人気。",
        "category": "洋食"
    },
    {
        "title": "オムライス",
        "description": "ケチャップライスとふわとろ卵の絶妙コンビ。おうちカフェにもぴったり。",
        "category": "洋食"
    },
    {
        "title": "豆腐とわかめの味噌汁",
        "description": "定番の和風スープ。食事のバランスを整えたいときに。",
        "category": "スープ"
    },
    {
        "title": "たまご雑炊",
        "description": "体調が悪いときや、夜食にぴったりのやさしい味。",
        "category": "あっさり"
    },
    {
        "title": "えびとアボカドのサンドイッチ",
        "description": "ぷりぷりのえびと濃厚なアボカドの相性が抜群なサンド。",
        "category": "軽食"
    },
    {
        "title": "じゃがいものガレット",
        "description": "外はカリッと中はもちっとした、フランス風のじゃがいも料理。",
        "category": "焼き物"
    },
    {
        "title": "ビーフストロガノフ",
        "description": "生クリームとマッシュルームが入ったコクのあるロシア風煮込み。",
        "category": "洋食"
    },
    {
        "title": "チキンのトマト煮込み",
        "description": "鶏肉とトマトでさっぱり煮込んだ栄養満点レシピ。パンにも合う！",
        "category": "煮込み料理"
    },
    {
        "title": "お好み焼き",
        "description": "キャベツたっぷり！ソースとマヨの最強コンボ。",
        "category": "粉もの"
    },
    {
        "title": "うどんカルボナーラ",
        "description": "和と洋の融合。もちもちうどんで作る濃厚カルボナーラ。",
        "category": "アレンジ"
    },
    {
        "title": "ごま豆乳鍋",
        "description": "クリーミーなごま豆乳スープで冬にぴったりの鍋。",
        "category": "鍋"
    },
    {
        "title": "さつまいもご飯",
        "description": "ほんのり甘くて秋に食べたい炊き込みご飯。",
        "category": "炊き込み"
    },
    {
        "title": "キャベツとツナの和風パスタ",
        "description": "和風だしであっさり仕上げた、野菜たっぷりのパスタです。",
        "category": "パスタ"
    }
]


# 📊 キャッシュファイル
CACHE_FILE = "recipe_vectors.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        recipe_cache = json.load(f)
else:
    recipe_cache = {}

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(recipe_cache, f)

# 🔄 ベクトル変換 + ハンドリング
def get_embedding(text: str, cache_key: str = None):
    if cache_key and cache_key in recipe_cache:
        return np.array(recipe_cache[cache_key])
    try:
        vec = model.encode(text).tolist()
        if cache_key:
            recipe_cache[cache_key] = vec
            save_cache()
        return np.array(vec)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

# 📬 ユーザー入力型
class MatchRequest(BaseModel):
    intent: str
    profile: str
    top_k: int = 3

@app.post("/match")
def match_recipes(req: MatchRequest):
    try:
        user_text = f"{req.intent}。{req.profile}"
        user_vec = get_embedding(user_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User embedding failed: {e}")

    results = []
    for recipe in recipes:
        try:
            recipe_vec = get_embedding(recipe["description"], cache_key=recipe["title"])
            similarity = util.cos_sim(user_vec, recipe_vec).item()
        except Exception as e:
            similarity = 0.0  # 失敗時は低スコアで代用
        results.append({
            "title": recipe["title"],
            "description": recipe["description"],
            "category": recipe["category"],
            "score": round(similarity, 3)
        })

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)[:req.top_k]
    return {"matches": sorted_results}
