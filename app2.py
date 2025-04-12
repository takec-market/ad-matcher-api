from sentence_transformers import SentenceTransformer, util
import pandas as pd

# モデルのロード（日本語対応・軽量）
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 仮のユーザー3人分
users = [
    {
        "name": "ユーザーA",
        "location": "東京都渋谷区",
        "intent": "外で体を動かしたい",
        "profile": "自然や運動が好きで、アウトドアが趣味"
    },
    {
        "name": "ユーザーB",
        "location": "大阪府梅田",
        "intent": "ゆっくりカフェで過ごしたい",
        "profile": "猫と静かな場所が好き。読書が趣味"
    },
    {
        "name": "ユーザーC",
        "location": "東京都新宿区",
        "intent": "今夜は遊びたい気分",
        "profile": "お酒が好きで、音楽と夜の街が大好き"
    }
]

# 広告10件（タイトル・主張文・提供エリア）
ads = [
    {"title": "朝ランイベント in 代々木公園", "message": "自然と運動が好きな朝活派向けのランニングイベントです", "area": "東京都渋谷区"},
    {"title": "ネコカフェ渋谷店", "message": "猫と静かに過ごしたい人のための癒やし空間です", "area": "東京都渋谷区"},
    {"title": "渋谷バー巡りガイド", "message": "ナイトライフを楽しみたい人向けのバー特集", "area": "東京都渋谷区"},
    {"title": "キャンプ用品セール", "message": "自然の中で過ごすのが好きな人に向けたアウトドア用品のセールです", "area": "全国"},
    {"title": "ビジネススーツ特売", "message": "オフィスで働く人に向けたスーツのディスカウントセール", "area": "全国"},
    {"title": "読書カフェ梅田店", "message": "静かに本が読めるカフェでリラックスした時間を提供します", "area": "大阪府梅田"},
    {"title": "猫カフェなんば", "message": "猫が好きな人向けの癒しのカフェです", "area": "大阪府なんば"},
    {"title": "クラブイベント新宿", "message": "音楽とお酒が好きな人向けのナイトクラブイベント", "area": "東京都新宿区"},
    {"title": "ヨガ体験イベント", "message": "自然と調和したい人のためのヨガ体験教室です", "area": "東京都中野区"},
    {"title": "ヘルシー弁当デリバリー", "message": "忙しい現代人の健康を考えた宅配弁当", "area": "全国"}
]

# マッチング処理
matched_results = []

for user in users:
    user_text = f"{user['intent']}。{user['profile']}"
    user_vec = model.encode(user_text, convert_to_tensor=True)

    results = []
    for ad in ads:
        ad_vec = model.encode(ad["message"], convert_to_tensor=True)
        similarity = util.cos_sim(user_vec, ad_vec).item()

        # エリア一致でボーナス
        area_bonus = 0.15 if user["location"] in ad["area"] else 0.0
        total_score = similarity + area_bonus

        results.append({
            "ユーザー": user["name"],
            "広告タイトル": ad["title"],
            "主張文": ad["message"],
            "エリア": ad["area"],
            "意味スコア": round(similarity, 3),
            "エリアボーナス": area_bonus,
            "合計スコア": round(total_score, 3)
        })

    # スコア順に並べて上位3件
    top_matches = sorted(results, key=lambda x: x["合計スコア"], reverse=True)[:3]
    matched_results.extend(top_matches)

# 結果をDataFrameで表示
df = pd.DataFrame(matched_results)
print(df.to_string(index=False))
