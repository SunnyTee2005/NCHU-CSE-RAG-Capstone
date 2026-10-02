"""
rag.py
======
RAG（Retrieval-Augmented Generation）的核心邏輯。

對外只暴露一個函式：ask(question: str) -> str

流程：
    1. 把使用者的問題轉成向量（embedding）
    2. 在 ChromaDB 中搜尋最相似的 5 個 chunks
       ├─ 若信心不足，再做關鍵字備援搜尋（針對課程名稱查詢）
    3. 組合成 prompt，呼叫 Gemini 生成答案
       （範圍外問題由 prompt 規則處理：找不到資料就回覆無法回答）

此檔案會被 app.py import 使用。
"""

import os
import re
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb

# ── 初始化（module-level：import 時執行一次，之後重複使用同一個連線）─────────

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("❌ 找不到 GOOGLE_API_KEY，請確認 .env 檔案。")

genai.configure(api_key=api_key)

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "nchu_cs_knowledge"
EMBEDDING_MODEL  = "models/gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"
TOP_K = 5  # 每次從資料庫撈幾個最相關的 chunk

# 向量搜尋信心門檻（cosine distance，越小越相似；超過此值視為「沒把握」）
LOW_CONFIDENCE_THRESHOLD = 0.5

# 建立 ChromaDB 連線（chroma_db/ 資料夾必須存在，即已跑過 ingest.py）
_client     = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_collection(name=COLLECTION_NAME)
_gen_model  = genai.GenerativeModel(GENERATION_MODEL)

# ── Step 1：將問題轉成向量 ─────────────────────────────────────────────────────

def _embed_query(text: str) -> list[float]:
    """
    把問題轉成向量。
    task_type="RETRIEVAL_QUERY" 代表這是「搜尋用的查詢」，
    和存入時用的 "RETRIEVAL_DOCUMENT" 是不同的最佳化方向。
    （這是 Google embedding API 的設計，可以提升搜尋準確度）
    """
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY"
    )
    return result["embedding"]

# ── Step 2：在向量資料庫搜尋最相關的 chunks ───────────────────────────────────

def _retrieve(query_embedding: list[float], year: str | None = None) -> list[dict]:
    """
    用向量相似度搜尋，找出最相關的 TOP_K 個 chunk。
    ChromaDB 會計算 query_embedding 和每個存入的 embedding 之間的 cosine 相似度。

    若指定 year，只搜尋該學年的資料 + 共用文件（year="all"），
    避免不同屆的規定混在一起干擾回答。
    """
    where_filter = None
    if year is not None:
        where_filter = {"$or": [{"year": year}, {"year": "all"}]}

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where=where_filter,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({"content": doc, "metadata": meta, "distance": dist})
    return chunks

# ── 備援：關鍵字搜尋（針對課程名稱查詢）────────────────────────────────────────

def _keyword_search_tables(question: str, year: str | None = None) -> list[dict]:
    """
    當向量搜尋信心不足時的備援機制。
    直接在所有 original_table metadata 中做字串比對，
    找出包含問題中課程名稱關鍵詞的表格 chunk。

    這能補足 AI 摘要未完整收錄課程名稱導致向量搜尋失敗的情況。
    """
    # 從資料庫撈出所有 chunk（含 metadata）
    all_results = _collection.get(include=["documents", "metadatas"])

    matched = []
    for doc, meta in zip(all_results["documents"], all_results["metadatas"]):
        original_table = meta.get("original_table", "")
        if not original_table:
            continue

        # 學年過濾
        if year is not None:
            chunk_year = meta.get("year", "all")
            if chunk_year != year and chunk_year != "all":
                continue

        # 字串比對：問題中的每個詞（兩字以上）是否出現在原始表格中
        tokens = [t for t in re.split(r"[\s，。？！、「」『』【】（）]", question) if len(t) >= 2]
        score = sum(1 for t in tokens if t in original_table)
        if score >= 1:
            matched.append({
                "content": doc,
                "metadata": meta,
                "distance": 0.3,  # 給一個固定的「中等信心」分數
                "from_keyword": True,
                "keyword_score": score,
            })

    # 依命中詞數排序，取最多 3 個
    matched.sort(key=lambda x: x["keyword_score"], reverse=True)
    return matched[:3]

# ── Step 3：組合 prompt ───────────────────────────────────────────────────────

def _build_prompt(question: str, retrieved_chunks: list[dict], year: str | None = None, history: list[dict] | None = None) -> str:
    """
    把撈回來的 chunks 組合成給 LLM 的 prompt。

    關鍵設計：
    - 對於「表格類型」的 chunk：
        向量搜尋時用的是 AI 摘要（content），
        但給 LLM 看的是「original_table」（完整原始表格）。
        這就是 Parent-Document Retrieval 的精髓。
    - 對於「文字類型」的 chunk：
        直接用 content。
    - history：最近幾輪的對話記錄，讓 Gemini 看懂代名詞（如「他」「這個」）。
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        source = chunk["metadata"].get("source", "未知來源")
        original_table = chunk["metadata"].get("original_table", "")

        if original_table:
            context_parts.append(
                f"[參考資料 {i}]（來源：{source}）\n{original_table}"
            )
        else:
            context_parts.append(
                f"[參考資料 {i}]（來源：{source}）\n{chunk['content']}"
            )

    context_text = "\n\n---\n\n".join(context_parts)

    year_instruction = ""
    if year is not None:
        year_instruction = f"\n5. 學生是 {year} 學年入學，若參考資料中有多個學年的規定，請優先依據 {year} 學年的資訊回答，並在答案中清楚標明學年。"

    # 組合對話歷史（用於理解代名詞、上下文）
    history_section = ""
    if history:
        history_lines = []
        for msg in history:
            role_label = "學生" if msg["role"] == "user" else "助手"
            history_lines.append(f"[{role_label}]：{msg['content']}")
        history_section = "\n=== 對話歷史（請用此理解代名詞與上下文，例如「他」「這門課」指的是什麼）===\n" + "\n\n".join(history_lines) + "\n"

    prompt = f"""你是國立中興大學資訊工程學系的智慧問答助手。
請根據以下提供的「參考資料」，以繁體中文回答學生的問題。
{history_section}
規則：
1. 只根據參考資料回答，不要捏造資料中沒有的資訊。
2. 如果參考資料中找不到答案，請直接說「抱歉，我目前的知識庫中找不到這個問題的答案，建議您直接詢問系辦或查看官方課程規章。」
3. 如果回答有依據，請在答案中標明來源，例如：「根據《111學年畢業條件》...」。
4. 回答請使用繁體中文，語氣親切友善。{year_instruction}

=== 參考資料 ===
{context_text}

=== 學生的問題 ===
{question}

=== 你的回答 ==="""

    return prompt

# ── 對外公開的主函式 ──────────────────────────────────────────────────────────

def ask(question: str, year: str | None = None, history: list[dict] | None = None) -> str:
    """
    RAG 主流程：輸入問題，回傳 AI 答案。

    改進項目：
    1. 低信心備援：向量搜尋結果信心不足時，補充關鍵字搜尋結果
    2. 多輪對話記憶：接收 history（前幾輪對話），讓 Gemini 看懂代名詞與上下文

    用法：
        from rag import ask
        answer = ask("資工系必修課有哪些？")
        answer = ask("我需要修幾學分？", year="113")
        answer = ask("那他是必修還是選修？", year="113", history=[...])  # 多輪對話
    """
    # ── 向量搜尋 ───────────────────────────────────────────────────────────────
    query_embedding  = _embed_query(question)
    retrieved_chunks = _retrieve(query_embedding, year=year)

    # ── 低信心備援：加入關鍵字搜尋結果 ────────────────────────────────────────
    # 若所有向量搜尋結果信心都不足，補充關鍵字搜尋
    best_distance = min((c["distance"] for c in retrieved_chunks), default=1.0)
    if best_distance > LOW_CONFIDENCE_THRESHOLD:
        keyword_chunks = _keyword_search_tables(question, year=year)
        if keyword_chunks:
            # 把關鍵字搜尋結果加到前面（優先讓 LLM 看到）
            retrieved_chunks = keyword_chunks + retrieved_chunks
            retrieved_chunks = retrieved_chunks[:TOP_K]  # 保持最多 TOP_K 個

    prompt   = _build_prompt(question, retrieved_chunks, year=year, history=history)
    response = _gen_model.generate_content(prompt)
    return response.text

# ── 直接執行時的測試 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        ("資料庫管理系統導論是必修嗎？", "112"),
        ("畢業最低需要幾個學分？", "113"),
        ("英文能力要達到什麼標準才能畢業？", None),
        ("我可以轉系嗎？", None),  # 範圍外問題：應回覆找不到答案
    ]

    for q, yr in test_questions:
        print(f"\n{'='*50}")
        print(f"問題：{q}（學年：{yr}）")
        print(f"{'='*50}")
        answer = ask(q, year=yr)
        print(f"回答：\n{answer}")
