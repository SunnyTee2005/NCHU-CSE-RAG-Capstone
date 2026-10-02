import os
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

input_filename = "gemini_output.md"

if not os.path.exists(input_filename):
    print(f"❌ 錯誤：找不到檔案 {input_filename}")
    exit(1)

# 讀取 gemini_output.md 內容
with open(input_filename, "r", encoding="utf-8") as f:
    markdown_document = f.read()

# 定義要作為切分依據的 Markdown 標題階層
# 這裡設定遇到單個井號 `#` 時，會將其視為 "Header 1" 來進行段落切分
# 遇到雙井號 `##` 時，會將其視為 "Header 2" 來進行切分
# 這樣做可以保留段落所屬的標題層級，對於後續套用 RAG (檢索增強生成) 會有很好的 Context 幫助
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

# 初始化 MarkdownHeaderTextSplitter，並將 strip_headers 設為 False (或是預設) 保留原本的 header 也可以
# 預設 strip_headers=True 會把 header 拿掉並放進 metadata 裡，這正是我們想要的
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 執行初次切分
md_header_splits = markdown_splitter.split_text(markdown_document)

# 二次切分：雖然 MarkdownHeaderTextSplitter 幫我們依照標題切分了段落並保留了 Context (Metadata)，
# 但某些標題底下的內容（例如超長的必修課表）可能還是太大，一旦超出 LLM 的 Context Window 限制就會報錯或失去焦點。
# 因此我們導入 RecursiveCharacterTextSplitter，限制每塊最多 400 字元，
# 並保留 50 字元的 Context Overlap (上下文重疊)，避免切斷重要的句子或表格邏輯。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# 使用 split_documents 將第一階段產出的 md_header_splits 進行二次切分
final_splits = text_splitter.split_documents(md_header_splits)

print(f"第一階段依據標題切分出 {len(md_header_splits)} 個 Chunk")
print(f"第二階段限制字數後，總共切分為 {len(final_splits)} 個 Chunk:\n" + "="*50)

# 將結果同時寫入檔案，方便檢視
output_file = "chunking_results.md"
with open(output_file, "w", encoding="utf-8") as out_f:
    out_f.write(f"# Chunking 測試結果\n\n- 第一階段 (標題切分): {len(md_header_splits)} 個 Chunk\n- 第二階段 (字數裁切): **总共切成了 {len(final_splits)} 個 Chunk**:\n\n---\n\n")
    
    for i, chunk in enumerate(final_splits):
        # 終端機印出
        print(f"--- Chunk {i+1} ---")
        print("【Metadata 標題資訊】:")
        print(chunk.metadata)
        print("\n【Page Content 內容】:")
        print(chunk.page_content.strip())
        print("="*50 + "\n")
        
        # 寫入檔案
        out_f.write(f"## Chunk {i+1}\n\n")
        out_f.write("### 【Metadata 標題資訊】:\n")
        out_f.write(f"```json\n{chunk.metadata}\n```\n\n")
        out_f.write("### 【Page Content 內容】:\n\n")
        out_f.write(f"{chunk.page_content.strip()}\n\n---\n\n")

print(f"✅ 測試完畢！結果已同步儲存至 {output_file}，請打開該檔案檢視。")
