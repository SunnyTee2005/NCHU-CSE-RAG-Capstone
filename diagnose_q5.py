"""
diagnose_q5.py
==============
診斷實驗裡唯一失分的 Q5：112 學年「資料庫管理系統導論」是不是必修？

背景：書面報告把失敗原因寫成「AI 摘要未完整列出科目名稱」，
      但檢查 all_chunks.json 會發現 chunk_14 的摘要裡確實有這個課名，
      所以報告的解釋不成立。這支腳本用來分出兩個候選原因：

  假設 A（語意稀釋）：摘要是 23 個課名的長清單，向量被稀釋，
                      目標表格根本沒進 TOP-5。
  假設 B（上下文流失）：撈到了，但表格被抽離 Markdown 時沒帶走章節標題，
                      整張表沒有「必修」二字，模型不敢斷言。

判讀：
  必修課表排名 > 5  → 假設 A 成立（檢索沒撈到）
  必修課表排名 <= 5 → 假設 A 不成立，問題出在假設 B

執行：  python diagnose_q5.py
需要：  .env 裡有可用的 GOOGLE_API_KEY，且 chroma_db/ 存在
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("找不到 GOOGLE_API_KEY，請確認 .env")
genai.configure(api_key=api_key)

QUESTION = "112 學年度入學學生，「資料庫管理系統導論」是不是必修？幾學分？"
TARGET   = "資料庫管理系統導論"
WHERE    = {"$or": [{"year": "112"}, {"year": "all"}]}

# ── 1. 用跟 rag.py 完全相同的方式把問題轉成向量 ──────────────────────────
qe = genai.embed_content(
    model="models/gemini-embedding-001",
    content=QUESTION,
    task_type="RETRIEVAL_QUERY",
)["embedding"]
print(f"向量維度：{len(qe)}   （報告寫 768 是錯的，實際應為 3072）\n")

col = chromadb.PersistentClient(path="./chroma_db").get_collection("nchu_cs_knowledge")

# ── 2. 實際的 TOP-5（rag.py 真正會看到的東西）────────────────────────────
res = col.query(
    query_embeddings=[qe], n_results=5, where=WHERE,
    include=["documents", "metadatas", "distances"],
)

print("=" * 66)
print("實際的 TOP-5")
print("=" * 66)
for i, (doc, meta, dist) in enumerate(
    zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), 1
):
    tbl = meta.get("original_table", "")
    hit = "★ 含目標課名" if TARGET in (tbl + doc) else "  —"
    kind = "表格" if tbl else "文字"
    print(f"{i}. dist={dist:.4f}  [{kind}] {meta.get('source')}  {hit}")
    print(f"     {doc[:64].replace(chr(10), ' ')}...")

best = min(res["distances"][0])
print(f"\n最佳 distance = {best:.4f}   低信心門檻 = 0.5")
print(f"會觸發關鍵字備援嗎？ {best > 0.5}")

# ── 3. 把候選池全撈出來，看必修課表到底排第幾 ────────────────────────────
n_all = col.count()
res2 = col.query(
    query_embeddings=[qe], n_results=n_all, where=WHERE,
    include=["metadatas", "distances"],
)
pool = len(res2["distances"][0])

print("\n" + "=" * 66)
print("判讀")
print("=" * 66)
found = False
for rank, (meta, dist) in enumerate(
    zip(res2["metadatas"][0], res2["distances"][0]), 1
):
    if TARGET in meta.get("original_table", ""):
        found = True
        print(f"必修課表在候選池中排名第 {rank} / {pool}，distance = {dist:.4f}")
        if rank > 5:
            print("→ 排在 TOP-5 之外：假設 A（語意稀釋、檢索沒撈到）成立。")
        else:
            print("→ 有進 TOP-5：假設 A 不成立，問題出在假設 B")
            print("  （表格沒帶章節標題，整張表沒有『必修』二字，模型不敢斷言）。")
        break
if not found:
    print("候選池裡找不到含目標課名的表格 chunk —— 檢查 year 過濾條件或知識庫內容。")

print("\n把上面的數字抄下來，面試時可以直接引用。")
