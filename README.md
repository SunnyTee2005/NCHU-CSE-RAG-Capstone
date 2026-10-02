# 中興資工智慧問答助手

**[English](README.en.md)**

以 RAG（Retrieval-Augmented Generation，檢索增強生成）技術打造的問答系統，讓國立中興大學資工系學生用自然語言查詢畢業學分、必修課程、英文能力畢業標準等課規資訊。系統回答時會標明依據的官方文件。

在 10 題標準測試中，本系統得分 **18/20（90%）**，未使用知識庫的純 Gemini 對照組僅 **2/20（10%）**。

![Demo](docs/images/demo_2.png)

## 為什麼需要這個系統

系上的畢業條件與課規分散在多份 PDF 裡，每個入學學年的規定都不同，而且大量資訊放在雙欄排版和表格中。直接問通用的大型語言模型，常會得到捏造的數字；本系統改為先從官方文件檢索相關段落，再讓模型根據這些段落作答。

## 系統架構

| 離線建庫 | 線上問答 |
|---|---|
| ![Offline indexing pipeline](docs/images/architecture_offline.png) | ![Online query pipeline](docs/images/architecture_online.png) |

**離線建庫（`run_pipeline.py` → `ingest.py`）**

1. 用 Gemini Vision 把 PDF 轉成 Markdown，處理一般 PDF 解析器容易出錯的雙欄排版與表格。
2. 純文字段落先依 Markdown 標題切分，再依字數切成 chunk（400 字，重疊 50 字）。
3. 每張表格請 Gemini 產生一段摘要，摘要用於向量搜尋，原始表格存在 metadata。
4. 以 `gemini-embedding-001` 產生向量，存入 ChromaDB（cosine 相似度）。

**線上問答（`rag.py`、`app.py`）**

1. 將問題向量化，在 ChromaDB 取回最相似的 5 個 chunk。
2. 如果使用者選了入學學年，只搜尋該學年與各學年共用的文件。
3. 向量搜尋信心不足時，改用關鍵字比對原始表格作為備援。
4. 組合 prompt 交給 Gemini 2.5 Flash。表格 chunk 提供的是完整原始表格，不是摘要。
5. Streamlit 介面保留最近 3 輪對話，讓模型理解「它」「這門課」等指代。

## 主要設計

- **Parent-Document Retrieval**：表格直接做 embedding 的效果不好，所以搜尋時比對 AI 摘要，回答時提供原始表格。摘要負責「找得到」，原始表格負責「答得準」。
- **學年過濾**：各學年的必修學分不同。每個 chunk 帶有 `year` 標籤，查詢時以 metadata 過濾，避免不同學年的規定混在一起。
- **關鍵字備援**：課程名稱這類精確字串，在語意向量中容易被稀釋。最佳距離超過門檻時，會對原始表格做字串比對補充結果。

## 實驗結果

對照組為純 Gemini（不提供任何文件），實驗組為本系統。10 題涵蓋 111–114 學年的學分、必修課程與英文標準，每題 0–2 分。

| 組別 | 得分 | 正確率 |
|---|---|---|
| 對照組（純 Gemini） | 2 / 20 | 10% |
| 實驗組（本系統） | 18 / 20 | 90% |

唯一失分的是 Q5（112 學年「資料庫管理系統導論」是否為必修）。`diagnose_q5.py` 用來區分兩種可能原因：摘要向量被長課程清單稀釋，或表格抽離後失去了章節標題的上下文。

實驗設計與完整分析見 [競賽報告](docs/competition_report.pdf)。

## 快速開始

需要 Python 3.10 以上，以及一組 [Google AI Studio](https://aistudio.google.com/) 的 API key。

```bash
git clone <this-repo-url>
cd nchu-cse-rag-assistant
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 在 .env 填入 GOOGLE_API_KEY
```

### 準備資料

本 repo **不附**系上的原始 PDF 與向量資料庫。請自行取得以下文件，放在 repo 根目錄，檔名需與 `run_pipeline.py` 中的 `PDF_CONFIGS` 一致：

| 檔名 | 學年標籤 |
|---|---|
| `111學年 畢業條件明細.pdf` | 111 |
| `112學年 資工系課規.pdf` | 112 |
| `113學年 畢業條件明細.pdf` | 113 |
| `114學年 畢業條件明細.pdf` | 114 |
| `國立中興大學學生英文能力畢業標準檢定辦法.pdf` | all |
| `國立中興大學通識英文課程修課暨抵免要點.pdf` | all |
| `RAG專題-額外的課規問題＆解答.pdf` | all |

找不到的檔案會被跳過。若要套用到其他系所，修改 `PDF_CONFIGS` 即可。

### 執行

```bash
python run_pipeline.py    # PDF → Markdown → chunks，輸出 all_chunks.json
python ingest.py          # chunks → 向量 → chroma_db/
streamlit run app.py      # 開啟 http://localhost:8501
```

其他腳本：

```bash
python experiment.py      # 對照實驗，結果寫入 experiment_results.md
python diagnose_q5.py     # Q5 失分診斷
python check_models.py    # 列出 API key 可用的 Gemini 模型
```

## 專案結構

```
.
├── app.py              Streamlit 聊天介面
├── rag.py              檢索與生成核心，對外提供 ask()
├── run_pipeline.py     離線前處理：PDF → Markdown → chunks
├── ingest.py           建立 ChromaDB 向量資料庫
├── experiment.py       RAG vs 純 LLM 對照實驗
├── diagnose_q5.py      失分題診斷
├── check_models.py     查詢可用模型
├── legacy/             早期單一 PDF 原型（已被上方流程取代）
└── docs/
    ├── competition_report.pdf
    └── images/
```

## 技術

Python · Google Gemini 2.5 Flash · gemini-embedding-001 · ChromaDB · LangChain Text Splitters · Streamlit

## 限制

- 答案品質取決於 PDF 解析與表格摘要的正確性，官方規定仍應以系上公告為準。
- 知識庫目前只涵蓋 111–114 學年與上述 7 份文件。
- 範圍外問題由 prompt 指示模型回覆「找不到答案」，不是獨立的分類器。

## 團隊

國立中興大學資訊工程學系 資訊專題（2025–2026）

- 鄭仲軒：系統設計、程式開發、實驗執行
- 隊友（姓名不公開）：文獻整理、實驗題目設計、進度監督
