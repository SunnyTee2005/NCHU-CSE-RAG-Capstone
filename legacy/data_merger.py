import json
import re
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def main():
    # 1. 讀取原始的乾淨 Markdown 檔
    with open("gemini_output.md", "r", encoding="utf-8") as f:
        md_text = f.read()

    # 2. 讀取你剛用 LLM 產生出來的表格摘要檔
    with open("processed_tables.json", "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    # 3. 【防呆機制】：捨棄 Regex，改用精準字串比對替換，確保表格殘骸不會混入純文字 Chunk
    text_without_tables = md_text
    for tb in tables_data:
        original_table = tb["original_table"]
        text_without_tables = text_without_tables.replace(original_table, '\n\n[表格已移至 metadata 處理]\n\n')

    # 4. 對「純文字」進行二次切分 (Hybrid Chunking)
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 先依標題切，再依字數切
    md_header_splits = markdown_splitter.split_text(text_without_tables)
    text_chunks = text_splitter.split_documents(md_header_splits)

    # 5. 【資料組裝】：統一所有資料的格式
    final_data = []

    # 5.1 塞入純文字 Chunks
    for i, chunk in enumerate(text_chunks):
        final_data.append({
            "id": f"text_chunk_{i}",
            "type": "text",
            "content": chunk.page_content, # 這是要給 ChromaDB 算向量跟搜尋的內容
            "metadata": chunk.metadata     # 這是標題層級，留著給 LLM 參考
        })

    # 5.2 塞入表格摘要與原始表格
    for i, tb in enumerate(tables_data):
        final_data.append({
            "id": f"table_chunk_{i}",
            "type": "table",
            "content": tb["summary"], # 讓 ChromaDB 搜這個 LLM 寫的密集關鍵字摘要
            "metadata": {
                "original_table": tb["original_table"], # 把真正的大表格藏在這裡
                "description": "這是從 PDF 萃取出的結構化表格"
            }
        })

    # 6. 輸出給隊友接手的最終檔案
    with open("final_ingestion_data.json", "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"✅ 資料整合完成！共產出 {len(text_chunks)} 筆純文字，與 {len(tables_data)} 筆表格。")
    print("👉 請將 'final_ingestion_data.json' 交接給負責資料庫的隊友。")

if __name__ == "__main__":
    main()