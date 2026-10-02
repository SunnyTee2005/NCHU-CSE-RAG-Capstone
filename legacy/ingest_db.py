import json
import chromadb

# ==========================================
# 步驟一：讀取準備好的 JSON 檔案
# ==========================================
print("📂 正在讀取 final_ingestion_data.json...")
# 確保使用 utf-8 編碼讀取，避免中文字變成亂碼
with open("final_ingestion_data.json", "r", encoding="utf-8") as file:
    data_list = json.load(file)

print(f"✅ 成功讀取資料，共有 {len(data_list)} 筆準備匯入！")

# ==========================================
# 步驟二：初始化本地端的 ChromaDB
# ==========================================
print(" 正在初始化 ChromaDB...")
# 指定一個資料夾來存放你的資料庫 (例如命名為 my_vector_db)
client = chromadb.PersistentClient(path="./my_vector_db")

# 建立或獲取一個 Collection (你可以自己取個喜歡的名字)
collection = client.get_or_create_collection(name="my_knowledge_base")


# ==========================================
# 步驟三：整理要寫入的資料列表
# ==========================================
# ChromaDB 喜歡我們一次把所有資料整理成三個大列表再寫入
documents = []
metadatas = []
ids = []

print("🔄 正在處理並轉換資料格式...")
# 使用 enumerate 可以同時取得資料的「索引數字」(i) 和「內容」(item)
for i, item in enumerate(data_list):
    
    # 1. 抓取文本 (準備放進 documents)
    # 使用 .get("content", "") 的好處是，如果剛好這筆資料沒有 content 欄位，它會回傳空字串而不會報錯
    content = item.get("content", "")
    
    # 2. 抓取元資料 (準備放進 metadatas)
    # 為了避免動到原始資料，我們複製一份來處理
    meta = item.get("metadata", {}).copy()

    if len(meta) == 0:
        meta["source"] = "json_file"
    
    # 【關鍵防錯機制】：檢查 meta 裡面的每一個值，如果不是純文字或數字，就轉成字串！
    for key, value in meta.items():
        # 如果 value 是字典(dict)或列表(list)這種複雜結構，例如你的「原始大表格」
        if isinstance(value, (dict, list)):
            # 將其轉換為 JSON 格式的字串 (ensure_ascii=False 確保中文正常顯示)
            meta[key] = json.dumps(value, ensure_ascii=False)
        elif value is None:
            # ChromaDB 也不接受 None，如果是 None 就轉成空字串
            meta[key] = ""
            
    # 3. 準備獨一無二的 ID (例如: doc_0, doc_1, doc_2...)
    doc_id = f"doc_{i}"
    
    # 把處理好的這筆資料，分別加進三個大列表中
    documents.append(content)
    metadatas.append(meta)
    ids.append(doc_id)


# ==========================================
# 步驟四：正式寫入 ChromaDB
# ==========================================
print("🚀 開始將資料寫入 ChromaDB (若資料量大可能需要幾分鐘)...")

# 使用 upsert (Update + Insert)：如果 ID 不存在就新增，如果 ID 已經存在就覆蓋更新。
# 這樣就算你不小心執行這支程式兩次，資料庫也不會出現重複的垃圾資料！
collection.upsert(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print("🎉 恭喜！資料匯入完成！你的專屬本地資料庫已經建置完畢。")