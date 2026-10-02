# legacy/

早期原型：只處理單一 PDF（112 學年課規），每一步拆成獨立腳本，向量資料庫存在 `my_vector_db/`。
後來整合為根目錄的 `run_pipeline.py` 與 `ingest.py`，並加入多學年支援。保留在這裡作為開發過程的紀錄。

| 腳本 | 步驟 |
|---|---|
| `extract_pdf.py` | 以 pdfplumber 解析 PDF（雙欄與表格效果不佳，因此改用 Gemini Vision） |
| `gemini_pdf_parser.py` | 以 Gemini Vision 將 PDF 轉為 Markdown |
| `table_summarizer.py` | 抽出表格並產生 AI 摘要 |
| `test_chunking.py` | 測試 Markdown 標題切分 + 字數切分 |
| `data_merger.py` | 合併文字與表格 chunks |
| `ingest_db.py` | 寫入 ChromaDB（使用預設 embedding） |
| `retrieve_db.py` | 測試檢索 |

腳本使用相對路徑，請在 repo 根目錄執行，例如 `python legacy/gemini_pdf_parser.py`。
`extract_pdf.py` 另需 `pip install pdfplumber==0.5.28`。
