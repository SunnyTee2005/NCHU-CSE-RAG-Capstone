"""
app.py
======
中興資工智慧問答助手的 Streamlit 聊天介面。

啟動方式：
    streamlit run app.py

注意：執行前請確認已完成以下兩個步驟：
    1. python run_pipeline.py  （處理 PDF，產生 all_chunks.json）
    2. python ingest.py        （建立向量資料庫 chroma_db/）
"""

import streamlit as st

# ── 頁面設定（必須是第一個 Streamlit 指令）──────────────────────────────────

st.set_page_config(
    page_title="中興資工 智慧問答助手",
    page_icon="🎓",
    layout="centered"
)

# ── 載入 RAG 模組 ──────────────────────────────────────────────────────────────

try:
    from rag import ask
except Exception as e:
    st.error(
        f"⚠️ 無法載入 RAG 模組：{e}\n\n"
        "請確認你已經依序執行：\n"
        "1. `python run_pipeline.py`\n"
        "2. `python ingest.py`"
    )
    st.stop()

# ── 歡迎訊息（每個新對話的第一條）──────────────────────────────────────────────

def _welcome_message() -> dict:
    return {
        "role": "assistant",
        "content": (
            "你好！我是中興資工的智慧問答助手 🎓\n\n"
            "我可以回答以下相關問題：\n"
            "- 📚 **畢業學分與必修課程**（如：必修課有哪些？最低學分是幾學分？）\n"
            "- 🌐 **英文能力畢業標準**（如：要考什麼證書？標準是什麼？）\n"
            "- ❓ **課規常見問題**（如：輔系怎麼修？外系課可以算學分嗎？）\n\n"
            "📅 **目前涵蓋：111、112、113、114 學年度的畢業條件**\n\n"
            "**建議先在左側選擇你的入學學年**，這樣我可以優先提供你那一屆的規定，避免不同年份的資訊混淆。\n\n"
            "請問有什麼需要幫忙的嗎？"
        )
    }

# ── 初始化多對話記錄 ──────────────────────────────────────────────────────────
#
# 為什麼用 session_state？
# Streamlit 每次使用者輸入就會把整個 app.py 從頭到尾重跑一遍。
# session_state 是 Streamlit 提供的「跨 rerun 的記憶空間」，
# 對話記錄存在這裡才不會每次都被清空。
#
# 注意：session_state 只在同一個瀏覽器標籤頁存活，重新整理後會消失。

if "conversations" not in st.session_state:
    st.session_state.conversations = [
        {"id": 0, "name": "新對話", "messages": [_welcome_message()]}
    ]
    st.session_state.current_conv_idx = 0
    st.session_state.next_conv_id = 1

def _current_conv() -> dict:
    """取得目前選中的對話。"""
    return st.session_state.conversations[st.session_state.current_conv_idx]

# ── 側邊欄：學年選擇 ＋ 對話管理 ─────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 設定")
    year_options = {
        "111 學年（大四）": "111",
        "112 學年（大三）": "112",
        "113 學年（大二）": "113",
        "114 學年（大一）": "114",
        "不確定／跨年級查詢": None,
    }
    selected_label = st.selectbox(
        "我的入學學年",
        options=list(year_options.keys()),
        index=4,  # 預設「不確定」
        help="選擇你的入學學年，系統會優先顯示該年度的畢業規定。"
    )
    selected_year = year_options[selected_label]

    if selected_year:
        st.info(f"目前查詢範圍：{selected_year} 學年畢業條件 ＋ 共用規章（英文能力等）")
    else:
        st.info("目前查詢範圍：所有學年（111–114）")

    st.divider()

    # ── 對話列表 ─────────────────────────────────────────────────────────────
    st.subheader("💬 對話記錄")

    # 新增對話按鈕
    if st.button("➕ 新對話", use_container_width=True):
        new_id = st.session_state.next_conv_id
        st.session_state.conversations.append({
            "id": new_id,
            "name": "新對話",
            "messages": [_welcome_message()]
        })
        st.session_state.current_conv_idx = len(st.session_state.conversations) - 1
        st.session_state.next_conv_id += 1
        st.rerun()

    # 列出所有對話，目前選中的會有「▶」標記
    for i, conv in enumerate(st.session_state.conversations):
        is_current = (i == st.session_state.current_conv_idx)
        label = f"▶ {conv['name']}" if is_current else conv["name"]
        if st.button(label, key=f"conv_btn_{conv['id']}", use_container_width=True):
            st.session_state.current_conv_idx = i
            st.rerun()

# ── 頁面標題 ───────────────────────────────────────────────────────────────────

st.title("🎓 中興資工 智慧問答助手")
st.caption("針對資工系畢業條件、課規、英文能力標準等問題，提供 AI 輔助查詢。涵蓋 111–114 學年度。資料以官方 PDF 為準。")

# ── 顯示目前對話的所有歷史訊息 ────────────────────────────────────────────────

for message in _current_conv()["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── 處理使用者輸入 ─────────────────────────────────────────────────────────────

if user_input := st.chat_input("請輸入你的問題..."):
    messages = _current_conv()["messages"]

    # 若這是對話裡第一個使用者問題，用它來命名這則對話（最多 20 字）
    user_msg_count = sum(1 for m in messages if m["role"] == "user")
    if user_msg_count == 0:
        short_name = user_input[:20] + ("…" if len(user_input) > 20 else "")
        st.session_state.conversations[st.session_state.current_conv_idx]["name"] = short_name

    # 1. 顯示使用者的訊息，並存入記錄
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 準備對話歷史（最近 3 輪 = 最多 6 條，讓 Gemini 理解代名詞和上下文）
    #    排除剛剛加入的使用者問題本身（messages[:-1]），避免重複傳送
    history = messages[:-1]   # 目前問題之前的所有訊息
    history = history[-6:]    # 只取最近 6 條（3 輪對話）

    # 3. 呼叫 RAG，顯示助手回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                answer = ask(user_input, year=selected_year, history=history)
            except Exception as e:
                answer = f"⚠️ 發生錯誤：{e}\n\n請確認 API Key 是否有效，以及向量資料庫是否已建立。"
        st.markdown(answer)

    # 4. 把助手回答存入記錄（下次 rerun 時重新顯示）
    messages.append({"role": "assistant", "content": answer})
