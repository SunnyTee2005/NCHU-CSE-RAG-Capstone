import os
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai

def process_markdown_tables(input_file, output_file):
    # 讀取資料：讀取 gemini_output.md
    if not os.path.exists(input_file):
        print(f"找不到檔案 {input_file}")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # ==========================================
    # 實作 Parent-Document Retrieval 概念：用摘要做向量搜尋，用原表做回答
    # ==========================================
    # 提取表格 (Regex)：使用 Regular Expression 找出檔案中所有的 Markdown 表格區塊（特徵為包含 |---| 的連續行）
    # 使用安全的 Regex 抓出所有包含 '|' 的連續行區塊，避免回溯導致無窮迴圈
    pattern = re.compile(r'(?:^.*\|.*\n?)+', re.MULTILINE)
    blocks = pattern.findall(content)
    
    # 過濾出含有類似 |---| 分隔線特徵的區段
    raw_tables = [b for b in blocks if re.search(r'\|[\-\s\:]+\|', b)]
    tables = [t.strip() for t in raw_tables if t.strip()]

    print(f"共提取到 {len(tables)} 個表格。")
    if not tables:
        return

    # 讀取 .env 載入 GOOGLE_API_KEY
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("錯誤：找不到 GOOGLE_API_KEY，請確認 .env 檔案中的設定。")
        return

    # 初始化 gemini-2.5-flash
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt_template = (
        "你是一個 RAG 資料庫的索引建立助手。請閱讀以下表格，產生一段用於『向量相似度檢索』的摘要。\n"
        "規則：\n"
        "1. 第一句必須點出表格的屬性（如：中興資工系 112 學年 61 學分必修課表）。\n"
        "2. 絕對不可省略任何實體名詞（Entity）。你必須將表格中所有的『科目名稱』完整、一字不漏地列在摘要中。\n"
        "3. 不需要優美的文句，越精煉越好。\n\n"
        "表格內容：\n{table}"
    )

    results = []

    # ==========================================
    # 實作 Parent-Document Retrieval 概念：用摘要做向量搜尋，用原表做回答
    # ==========================================
    # LLM 摘要生成：對每一個抓出來的表格呼叫 API
    for i, table in enumerate(tables, 1):
        print(f"正在為第 {i}/{len(tables)} 個表格生成摘要...")
        prompt = prompt_template.format(table=table)
        
        try:
            response = model.generate_content(prompt)
            summary = response.text.strip()
            
            # 資料結構化與儲存
            results.append({
                "summary": summary,
                "original_table": table
            })
            print(f"✅ 第 {i} 個表格摘要完成。")
            
        except Exception as e:
            print(f"❌ 第 {i} 個表格摘要發生錯誤：{e}")

    # 將此 JSON 陣列存檔為 processed_tables.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n處理完畢，已將包含 {len(results)} 個物件的 JSON 陣列結果儲存至 {output_file}。")

if __name__ == "__main__":
    input_filepath = "gemini_output.md"
    output_filepath = "processed_tables.json"
    process_markdown_tables(input_filepath, output_filepath)
