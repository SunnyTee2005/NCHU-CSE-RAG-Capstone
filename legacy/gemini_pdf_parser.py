import os
from dotenv import load_dotenv
import google.generativeai as genai
import time

# 1. 載入環境變數
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ 錯誤：找不到 API Key！請檢查你的 .env 檔案。")
    exit(1)

# 2. 初始化 Gemini API 並且指定模型
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

pdf_filename = "112學年 資工系課規.pdf"
output_filename = "gemini_output.md"

if not os.path.exists(pdf_filename):
    print(f"❌ 錯誤：找不到檔案 {pdf_filename}")
    exit(1)

# 3. 上傳檔案至 Gemini File API (避免 base64 OOM 問題)
print(f"上傳檔案: {pdf_filename} ...")
uploaded_file = genai.upload_file(pdf_filename)
print(f"上傳成功! 檔案 URI: {uploaded_file.uri}")

# 等待檔案後台處理
print("等待檔案準備就緒...")
while uploaded_file.state.name == "PROCESSING":
    time.sleep(2)
    uploaded_file = genai.get_file(uploaded_file.name)
    
if uploaded_file.state.name == "FAILED":
    print("❌ 檔案處理失敗。")
    exit(1)

# 利用 Vision LLM 解決傳統 OCR 無法處理雙欄排版與隱形表格的痛點
prompt = "你是一個專業的資料處理工程師。請閱讀這份雙欄排版的系所規章 PDF，並將所有內容轉換為乾淨、標準的 Markdown 格式。特別注意：必須精準還原所有跨頁或隱含的表格（如必修課表、學分表），將其強制轉為 Markdown 表格語法（|---|），確保欄位對齊、無文字錯位或遺漏。請直接輸出 Markdown 內容，不需要任何開場白。"

print("🤖 呼叫 Gemini 進行多模態解析...")
try:
    response = model.generate_content([uploaded_file, prompt])
    
    # 4. 輸出解析結果至 markdown 檔案
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(response.text)
        
    print(f"🎉 解析完成！結果已儲存至 {output_filename}")
    
except Exception as e:
    print(f"❌ 發生錯誤: {e}")
