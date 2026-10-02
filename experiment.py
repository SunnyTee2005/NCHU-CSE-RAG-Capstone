"""
experiment.py
=============
實驗腳本：對照組 vs 實驗組

對照組：純 Gemini，不給任何文件，不使用 RAG
實驗組：完整 RAG 系統（from rag import ask）

執行方式：
    python experiment.py

結果會存到 experiment_results.md，方便對照標準答案打分。
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
from rag import ask as rag_ask

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ── 測試題目（含對應的入學學年，給 RAG 用來過濾資料）─────────────────────────

QUESTIONS = [
    {
        "id": 1,
        "question": "111 學年度入學學生，最低畢業總學分數是多少？",
        "year": "111",
        "standard_answer": "共 128 學分（不含體育課程）",
    },
    {
        "id": 2,
        "question": "111 學年度入學學生，系專業必修課程最低要修幾學分？",
        "year": "111",
        "standard_answer": "最低應修 64 學分",
    },
    {
        "id": 3,
        "question": "111 學年度入學學生，外系課程加上相關院系課程與體育課，最多可以採計幾學分？",
        "year": "111",
        "standard_answer": "至多 18 學分",
    },
    {
        "id": 4,
        "question": "112 學年度入學學生，系專業必修課程最低要修幾學分？",
        "year": "112",
        "standard_answer": "最低應修 61 學分",
    },
    {
        "id": 5,
        "question": "112 學年度入學學生，「資料庫管理系統導論」是不是必修？幾學分？",
        "year": "112",
        "standard_answer": "必修，3 學分",
    },
    {
        "id": 6,
        "question": "112 學年度入學學生，資訊專題（三）能不能拿來抵資訊專題（一）或（二）？",
        "year": "112",
        "standard_answer": "可以，資訊專題（三）可抵資訊專題（一）或資訊專題（二）",
    },
    {
        "id": 7,
        "question": "113 學年度入學學生，系專業必修課程最低要修幾學分？",
        "year": "113",
        "standard_answer": "最低應修 55 學分",
    },
    {
        "id": 8,
        "question": "114 學年度入學學生，外國語文 6 學分包含哪三門課？",
        "year": "114",
        "standard_answer": "英語溝通與表達（2學分）、學術英文聽讀（2學分）、學術英文說寫（2學分）",
    },
    {
        "id": 9,
        "question": "通過學校英文能力畢業標準時，多益最低要幾分？",
        "year": None,
        "standard_answer": "多益（TOEIC）670 分以上",
    },
    {
        "id": 10,
        "question": "若想申請抵免「英語溝通與表達」及「學術英文聽讀」，英文檢定成績至少要達到哪些門檻？",
        "year": None,
        "standard_answer": "TOEFL iBT 72分、IELTS 5.5級、Cambridge FCE、TOEIC 785分、全民英檢中高級複試，或近三年曾在英語系國家求學兩年以上",
    },
]

# ── 對照組：純 Gemini，不給任何文件 ──────────────────────────────────────────

BASELINE_MODEL = genai.GenerativeModel("gemini-2.5-flash")

BASELINE_PROMPT_TEMPLATE = """你是一個通用 AI 助理。請用繁體中文回答以下問題。
請只根據你本身的知識回答，不要說你無法查詢網路。

背景：以下是關於國立中興大學資訊工程學系學士班的課程規定與畢業條件問題。

問題：{question}
"""

def ask_baseline(question: str) -> str:
    prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
    response = BASELINE_MODEL.generate_content(prompt)
    return response.text

# ── 主流程 ────────────────────────────────────────────────────────────────────

def run_experiment():
    results = []

    for q in QUESTIONS:
        qid      = q["id"]
        question = q["question"]
        year     = q["year"]

        print(f"\n{'='*60}")
        print(f"[Q{qid}] {question}")
        print(f"{'='*60}")

        # 對照組
        print("⏳ 對照組（純 Gemini）...")
        baseline_answer = ask_baseline(question)
        print(f"對照組回答：{baseline_answer[:200]}...")

        # 實驗組
        print("⏳ 實驗組（RAG）...")
        rag_answer = rag_ask(question, year=year)
        print(f"實驗組回答：{rag_answer[:200]}...")

        results.append({
            "id": qid,
            "question": question,
            "standard_answer": q["standard_answer"],
            "baseline_answer": baseline_answer,
            "rag_answer": rag_answer,
        })

    # 輸出成 Markdown 方便打分
    save_results(results)
    print("\n\n✅ 完成！結果已存到 experiment_results.md，請開啟後逐題打分。")

def save_results(results):
    lines = ["# 實驗結果\n"]
    lines.append("## 評分說明")
    lines.append("- **2 分**：完全正確，關鍵數字與條件都對")
    lines.append("- **1 分**：部分正確，方向對但漏了細節或有小錯")
    lines.append("- **0 分**：錯誤或完全沒有相關回答\n")
    lines.append("---\n")

    for r in results:
        lines.append(f"## Q{r['id']}：{r['question']}\n")
        lines.append(f"**標準答案：** {r['standard_answer']}\n")
        lines.append(f"### 對照組（純 Gemini，無 RAG）")
        lines.append(f"{r['baseline_answer']}\n")
        lines.append(f"**對照組分數：** ＿＿ / 2\n")
        lines.append(f"### 實驗組（RAG 系統）")
        lines.append(f"{r['rag_answer']}\n")
        lines.append(f"**實驗組分數：** ＿＿ / 2\n")
        lines.append("---\n")

    lines.append("## 總分統計\n")
    lines.append("| 組別 | 總分 | 滿分 |")
    lines.append("|---|---|---|")
    lines.append("| 對照組（純 Gemini） | ＿＿ | 20 |")
    lines.append("| 實驗組（RAG） | ＿＿ | 20 |")

    with open("experiment_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    run_experiment()
