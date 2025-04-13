FROM python:3.10-slim

# システム依存パッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# Python依存パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体コピー
COPY . .

# Cloud Run用ポート指定
ENV PORT=8080

# 起動コマンド（Cloud Runは 0.0.0.0:8080 を使う）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]