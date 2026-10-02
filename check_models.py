import os
from dotenv import load_dotenv
import google.generativeai as genai

# 1. 載入密碼
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("🔍 正在查詢你的帳號可用的模型列表...\n")

try:
    # 列出所有模型
    for m in genai.list_models():
        # 我們只關心能「產生內容 (generateContent)」的模型
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            
    print("\n✅ 查詢完成。請選一個上面的名字填回 app.py")

except Exception as e:
    print(f"❌ 查詢失敗：{e}")