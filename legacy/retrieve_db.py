import chromadb

# ==========================================
# 步驟一：連線到我們剛剛建好的本地端 ChromaDB
# ==========================================
print("🔌 正在連線至本地 ChromaDB...")
client = chromadb.PersistentClient(path="./my_vector_db")

# 獲取我們剛剛建立的 Collection
# (請確保名稱跟你在 ingest_db.py 裡面取的一模一樣)
collection = client.get_collection(name="my_knowledge_base")
print("✅ 資料庫連線成功！\n")

# 👇 請在這裡加上這行來檢查庫存！
print(f"📦 目前資料庫裡總共有 {collection.count()} 筆資料喔！")


# ==========================================
# 步驟二：封裝查詢 Function (核心邏輯)
# ==========================================
def retrieve_context(query, k=3):
    """
    接收使用者的問題，找出最相似的 Top-K 筆資料，
    並根據 metadata 決定要回傳「純文字」還是「原始大表格」。
    """
    print(f"🔍 正在搜尋問題：『{query}』")
    
    # 1. 透過 ChromaDB 找出最相似的 k 筆資料
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    # ChromaDB 回傳的資料是在一個二維陣列裡 (因為可以一次查多個 query)
    # 我們只查了一個 query，所以取 [0] 即可
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    final_contexts = []
    
    # 2. 核心邏輯：用 zip 把 document 和 metadata 一對一綁起來檢查
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        
        # 檢查 metadata 裡面有沒有 "original_table" 這個 Key，且內容不是空的
        if "original_table" in meta and meta["original_table"].strip() != "":
            # 【命中表格摘要】
            print(f"  👉 第 {i+1} 筆命中：表格摘要！(將抽取完整 Markdown 表格作為 Context)")
            # 我們不使用 doc (摘要)，而是把隱藏在 metadata 裡的大表格抽出來
            final_contexts.append(meta["original_table"])
            
        else:
            # 【命中純文字】
            print(f"  👉 第 {i+1} 筆命中：純文字條文！")
            # 直接使用原本的純文字內容
            final_contexts.append(doc)
            
    # 3. 把這 k 筆挑出來的 Context 用分隔線串接成一大段文字
    joined_context = "\n\n==== 分隔線 ====\n\n".join(final_contexts)
    
    return joined_context


# ==========================================
# 步驟三：測試我們的 Function
# ==========================================
if __name__ == "__main__":
    # 假設這是使用者輸入的問題 (你可以改成跟你 JSON 內容相關的問題)
    user_question = "請幫我列出包含23門科目的課程清單有哪些"
    
    # 呼叫我們寫好的 Function
    context_for_llm = retrieve_context(query=user_question, k=3)
    
    # 印出最終要準備餵給 LLM 的 Context
    print("\n📦 【最終準備給 LLM 的 Context 內容如下】:")
    print("-" * 50)
    print(context_for_llm)
    print("-" * 50)
