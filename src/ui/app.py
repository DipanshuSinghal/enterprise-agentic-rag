import streamlit as st
import requests
import json
import uuid  # 🟢 NEW: Import UUID to generate unique session keys

# 1. Configure page view dimensions and branding layouts
st.set_page_config(page_title="Enterprise AI Assistant", page_icon="🤖", layout="centered")
st.title("Enterprise AI Knowledge Assistant")
st.subheader("State-Driven Academic & Web Search Engine")

# 2. 🟢 NEW: Initialize a truly unique, persistent thread ID for this specific chat session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize safe persistent conversation arrays inside state memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Render historical conversation threads so they stay visible on refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📚 Verified References Used"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Reference [{idx}]:**")
                    st.info(src)

# 4. Capture inbound text from the chat entry input field
user_input = st.chat_input("Ask your knowledge base a question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()
        
        full_response = ""
        retrieved_sources = []
        api_url = "http://127.0.0.1:8000/api/v1/query"
        
        try:
            # 🟢 UPDATED: Pass both the user query and the unique thread_id to the backend payload
            payload = {
                "query": user_input,
                "thread_id": st.session_state.thread_id
            }
            
            response = requests.post(api_url, json=payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith("data: "):
                        json_str = decoded_line.replace("data: ", "").strip()
                        try:
                            data = json.loads(json_str)
                            if "node" in data:
                                status_placeholder.caption(f"⚙️ Node active: `{data['node'].upper()}`...")
                            if "text" in data and data["text"]:
                                full_response += str(data["text"])
                                response_placeholder.markdown(full_response + "▌")
                            if "sources" in data:
                                retrieved_sources = data["sources"]
                        except json.JSONDecodeError:
                            continue
                            
            response_placeholder.markdown(full_response)
            status_placeholder.empty()
            
            if retrieved_sources:
                with st.expander("📚 Verified References & Context Sources", expanded=False):
                    for idx, source in enumerate(retrieved_sources, 1):
                        st.markdown(f"**Source [{idx}]:**")
                        st.info(source)
            else:
                st.caption("✅ Execution finished (No external source references used).")
                
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": retrieved_sources
            })
            
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Streaming network error occurred: {str(e)}")