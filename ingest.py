"""
ingest.py
=========
讀取 all_chunks.json，將每個 chunk 轉成向量（embedding），存入本地 ChromaDB。

執行方式：
    python ingest.py

只需要執行一次。若想重建資料庫，直接再跑一次，它會自動清掉舊的再重建。

什麼是 Embedding？
    把一段文字轉成一個數字陣列（向量），讓電腦能計算兩段文字的「語意相似度」。
    例如：「微積分必修嗎？」和「微積分是必修課」語意相近 → 向量的夾角小 → cosine 相似度高。

什麼是 ChromaDB？
    一個向量資料庫，專門儲存和搜尋 embedding 向量，可以完全跑在本機（不需要外部伺服器）。
"""

import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# ── 初始化 ──────────────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ 錯誤：找不到 API Key！請確認 .env 檔案中有 GOOGLE_API_KEY=...")
    exit(1)

genai.configure(api_key=api_key)

# ── 設定 ──────────────────────────────────────────────────────────────────────

CHUNKS_FILE     = "all_chunks.json"    # run_pipeline.py 的輸出
CHROMA_PATH     = "./chroma_db"        # ChromaDB 儲存在本機的資料夾
COLLECTION_NAME = "nchu_cs_knowledge"
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBED_BATCH_SIZE = 20  # 每次批次送幾個給 API（Google API 有速率限制）

# ── 載入資料 ──────────────────────────────────────────────────────────────────

if not os.path.exists(CHUNKS_FILE):
    print(f"❌ 找不到 {CHUNKS_FILE}。請先執行：python run_pipeline.py")
    exit(1)

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"📂 載入 {len(chunks)} 個 chunks，開始建立向量資料庫...\n")

# ── 初始化 ChromaDB ────────────────────────────────────────────────────────────

# PersistentClient：資料存在硬碟，重啟後不會消失
client = chromadb.PersistentClient(path=CHROMA_PATH)

# 若舊的 collection 存在就刪掉（確保每次重建都是乾淨的狀態）
try:
    client.delete_collection(name=COLLECTION_NAME)
    print("🗑️  已刪除舊的資料集，重新建立...\n")
except Exception:
    pass  # 第一次執行，collection 不存在，直接跳過

# 建立新的 collection，用 cosine 相似度計算（比預設的 L2 距離更適合文字 embedding）
collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ── Embedding 輔助函式 ─────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批次將一串文字轉成向量。
    task_type="RETRIEVAL_DOCUMENT" 告訴 Google：這是要「存入資料庫」的文件。
    （搜尋時用的是 "RETRIEVAL_QUERY"，兩者最佳化方向不同）
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=texts,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return result["embedding"]

# ── 批次 Embedding 並存入 ChromaDB ────────────────────────────────────────────

for i in range(0, len(chunks), EMBED_BATCH_SIZE):
    batch = chunks[i : i + EMBED_BATCH_SIZE]
    batch_end = min(i + EMBED_BATCH_SIZE, len(chunks))

    texts = [c["content"] for c in batch]
    ids   = [c["id"] for c in batch]

    # ChromaDB 的 metadata 欄位只能是 str / int / float / bool
    # 若有 dict 等複雜型別（例如 original_table 是字串，沒問題），需要轉成 str
    metadatas = []
    for c in batch:
        safe_meta = {}
        for k, v in c["metadata"].items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)  # 防呆：轉成字串
        metadatas.append(safe_meta)

    print(f"  嵌入第 {i+1} ~ {batch_end} 個 chunk（共 {len(chunks)} 個）...")
    embeddings = embed_texts(texts)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

# ── 驗證 ──────────────────────────────────────────────────────────────────────

count = collection.count()
print(f"\n✅ 資料庫建立完成！共 {count} 筆資料已儲存至 {CHROMA_PATH}/")
print("👉 下一步：執行 streamlit run app.py 啟動聊天介面")
