FROM python:3.10-slim

# システムの更新と必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ作成
WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# アプリ本体をコピー
COPY . .

# 実行
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
