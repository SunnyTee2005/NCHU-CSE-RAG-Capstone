"""
run_pipeline.py
================
將所有 PDF 跑完整的資料前處理 pipeline，合併輸出成 all_chunks.json。

執行方式：
    python run_pipeline.py

流程：
    對每個 PDF：
        1. parse_pdf_to_markdown()   → PDF → Markdown（Gemini Vision API）
        2. summarize_tables()        → 提取表格 → AI 摘要（Parent-Document Retrieval）
        3. merge_chunks()            → 合併純文字 chunks + 表格 chunks
    最後合併所有 PDF 的結果 → all_chunks.json
"""

import os
import re
import json
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ── 初始化 ──────────────────────────────────────────────────────────────────

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ 錯誤：找不到 API Key！請確認 .env 檔案中有 GOOGLE_API_KEY=...")
    exit(1)

genai.configure(api_key=api_key)

PARSE_MODEL = genai.GenerativeModel("gemini-2.5-flash")
SUMMARIZE_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ── Step 1：PDF → Markdown ────────────────────────────────────────────────────

def parse_pdf_to_markdown(pdf_path: str, output_md_path: str):
    """
    上傳 PDF 至 Gemini File API，用 Vision LLM 解析成乾淨的 Markdown。
    解決傳統 pdfplumber 無法處理的雙欄排版、隱形表格問題。
    """
    print(f"  [1/3] 上傳 {pdf_path} 至 Gemini File API...")
    uploaded_file = genai.upload_file(pdf_path)

    # 等待 Gemini 後台處理（狀態從 PROCESSING → ACTIVE）
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise RuntimeError(f"❌ 檔案處理失敗：{pdf_path}")

    prompt = (
        "你是一個專業的資料處理工程師。請閱讀這份 PDF（可能是雙欄排版或含有表格的系所規章），"
        "並將所有內容轉換為乾淨、標準的 Markdown 格式。"
        "特別注意：必須精準還原所有表格（如必修課表、學分表），"
        "將其轉為 Markdown 表格語法（|---|），確保欄位對齊、無文字錯位或遺漏。"
        "請直接輸出 Markdown 內容，不需要任何開場白。"
    )

    print(f"  [1/3] 呼叫 Gemini 解析 PDF...")
    response = PARSE_MODEL.generate_content([uploaded_file, prompt])

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"  [1/3] ✅ Markdown 已儲存至 {output_md_path}")

# ── Step 2：Markdown 中的表格 → AI 摘要 ──────────────────────────────────────

def summarize_tables(md_path: str, tables_json_path: str) -> list:
    """
    從 Markdown 中提取所有表格，用 LLM 為每個表格生成密集摘要。
    回傳 list，格式：[{"summary": "...", "original_table": "..."}, ...]

    Parent-Document Retrieval 的精髓：
    - 向量資料庫搜尋「summary」（語意豐富）
    - 回答時提供「original_table」（完整資訊）
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 用安全的 Regex 找出所有包含 | 的連續行區塊（避免回溯造成無窮迴圈）
    pattern = re.compile(r"(?:^.*\|.*\n?)+", re.MULTILINE)
    blocks = pattern.findall(content)

    # 過濾出真正的表格（含有 |---| 分隔線）
    raw_tables = [b for b in blocks if re.search(r"\|[\-\s\:]+\|", b)]
    tables = [t.strip() for t in raw_tables if t.strip()]

    print(f"  [2/3] 共找到 {len(tables)} 個表格，開始生成摘要...")

    results = []
    prompt_template = (
        "你是一個 RAG 資料庫的索引建立助手。請閱讀以下表格，產生一段用於『向量相似度檢索』的摘要。\n"
        "規則：\n"
        "1. 第一句必須點出表格的屬性（如：中興資工系 112 學年 61 學分必修課表）。\n"
        "2. 絕對不可省略任何實體名詞（Entity）。你必須將表格中所有的『科目名稱』完整、一字不漏地列在摘要中。\n"
        "3. 不需要優美的文句，越精煉越好。\n\n"
        "表格內容：\n{table}"
    )

    for i, table in enumerate(tables, 1):
        print(f"  [2/3] 生成第 {i}/{len(tables)} 個表格的摘要...")
        try:
            response = SUMMARIZE_MODEL.generate_content(prompt_template.format(table=table))
            results.append({
                "summary": response.text.strip(),
                "original_table": table
            })
            # 避免觸發 API rate limit
            time.sleep(2)
        except Exception as e:
            print(f"  ⚠️  第 {i} 個表格摘要失敗（跳過）：{e}")

    with open(tables_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"  [2/3] ✅ 表格摘要已儲存至 {tables_json_path}")
    return results

# ── Step 3：合併純文字 + 表格 → chunks list ───────────────────────────────────

def merge_chunks(md_path: str, tables_data: list, source_name: str, year: str = "all") -> list:
    """
    將 Markdown 純文字 + 表格摘要合併成統一格式的 chunk list。
    每個 chunk 的 metadata 都會加上 source，讓 LLM 之後可以引用來源。

    回傳格式：
    [
        {"type": "text",  "content": "...", "metadata": {"source": "...", "Header 1": "..."}},
        {"type": "table", "content": "AI摘要", "metadata": {"source": "...", "original_table": "..."}},
    ]
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # 把表格從純文字中移除（用精確字串替換，避免 Regex 殘骸問題）
    text_without_tables = md_text
    for tb in tables_data:
        text_without_tables = text_without_tables.replace(
            tb["original_table"], "\n\n[表格已移至 metadata 處理]\n\n"
        )

    # Hybrid Chunking：先依標題切，再依字數切
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    md_header_splits = markdown_splitter.split_text(text_without_tables)
    text_chunks = text_splitter.split_documents(md_header_splits)

    chunks = []

    # 純文字 chunks
    for chunk in text_chunks:
        metadata = dict(chunk.metadata)  # 含 Header 1 / Header 2
        metadata["source"] = source_name  # 加上來源標籤
        metadata["year"] = year           # 加上年份標籤（用於 RAG 年份過濾）
        chunks.append({
            "type": "text",
            "content": chunk.page_content,
            "metadata": metadata
        })

    # 表格 chunks（用 AI 摘要做搜尋，把原始表格藏在 metadata）
    for tb in tables_data:
        chunks.append({
            "type": "table",
            "content": tb["summary"],
            "metadata": {
                "source": source_name,
                "year": year,
                "original_table": tb["original_table"]
            }
        })

    print(f"  [3/3] ✅ 共切出 {len(text_chunks)} 個文字 chunk、{len(tables_data)} 個表格 chunk")
    return chunks

# ── 主程式：對所有 PDF 跑 pipeline ───────────────────────────────────────────

# 設定要處理的 PDF 清單
PDF_CONFIGS = [
    {
        "pdf":    "111學年 畢業條件明細.pdf",
        "source": "111學年畢業條件",
        "year":   "111",
        "md":     "temp_grad_111.md",
        "tables": "temp_grad_111_tables.json",
    },
    {
        "pdf":    "112學年 資工系課規.pdf",
        "source": "112學年畢業條件",
        "year":   "112",
        "md":     "temp_cs_grad.md",
        "tables": "temp_cs_grad_tables.json",
    },
    {
        "pdf":    "113學年 畢業條件明細.pdf",
        "source": "113學年畢業條件",
        "year":   "113",
        "md":     "temp_grad_113.md",
        "tables": "temp_grad_113_tables.json",
    },
    {
        "pdf":    "114學年 畢業條件明細.pdf",
        "source": "114學年畢業條件",
        "year":   "114",
        "md":     "temp_grad_114.md",
        "tables": "temp_grad_114_tables.json",
    },
    {
        "pdf":    "國立中興大學學生英文能力畢業標準檢定辦法.pdf",
        "source": "英文能力畢業標準",
        "year":   "all",
        "md":     "temp_english.md",
        "tables": "temp_english_tables.json",
    },
    {
        "pdf":    "RAG專題-額外的課規問題＆解答.pdf",
        "source": "課規FAQ",
        "year":   "all",
        "md":     "temp_faq.md",
        "tables": "temp_faq_tables.json",
    },
    {
        "pdf":    "國立中興大學通識英文課程修課暨抵免要點.pdf",
        "source": "通識英文修課暨抵免要點",
        "year":   "all",
        "md":     "temp_english_exemption.md",
        "tables": "temp_english_exemption_tables.json",
    },
]

if __name__ == "__main__":
    all_chunks = []
    chunk_id_counter = 0  # 全域計數器，確保 ID 在所有 PDF 中唯一

    for config in PDF_CONFIGS:
        pdf_path = config["pdf"]
        print(f"\n{'='*50}")
        print(f"[處理中] {pdf_path}")
        print(f"{'='*50}")

        if not os.path.exists(pdf_path):
            print(f"  ⚠️  找不到 {pdf_path}，跳過。")
            continue

        # Step 1
        parse_pdf_to_markdown(pdf_path, config["md"])

        # Step 2
        tables_data = summarize_tables(config["md"], config["tables"])

        # Step 3
        chunks = merge_chunks(config["md"], tables_data, config["source"], config["year"])

        # 指派全域唯一 ID
        for chunk in chunks:
            chunk["id"] = f"chunk_{chunk_id_counter}"
            chunk_id_counter += 1

        all_chunks.extend(chunks)
        print(f"  累計共 {len(all_chunks)} 個 chunks")

    # 輸出最終合併結果
    output_path = "all_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*50}")
    print(f"✅ 全部完成！共 {len(all_chunks)} 個 chunks 已儲存至 {output_path}")
    print("👉 下一步：執行 python ingest.py 建立向量資料庫")
    print(f"{'='*50}")

    # 清理暫存檔案
    for config in PDF_CONFIGS:
        for path in [config["md"], config["tables"]]:
            if os.path.exists(path):
                os.remove(path)
    print("🧹 暫存檔案已清理完畢")
