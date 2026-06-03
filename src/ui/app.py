import streamlit as st
import requests

# 1. Page Configuration Customization
st.set_page_config(
    page_title="Enterprise Agentic RAG Chat",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Enterprise Agentic RAG Chatbot")
st.markdown("Interact live with your self-correcting LangGraph & Qdrant pipeline.")

# 2. Initialize chat message history state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Render all existing messages from history on page refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "steps" in message:
            st.caption(f"🧭 **Agent Routing Path:** { ' ➔ '.join(message['steps']) }")

# 4. Accept fresh user input
if user_query := st.chat_input("Ask your knowledge base a question..."):
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display assistant processing placeholder spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        with st.spinner("Agent thinking and verifying documents..."):
            try:
                # Fire standard POST request directly to your local FastAPI gateway server
                api_url = "http://127.0.0.1:8000/api/v1/query"
                payload = {"query": user_query}
                response = requests.post(api_url, json=payload, timeout=500)
                
                if response.status_code == 200:
                    data = response.json()
                    generation = data.get("generation", "No generation produced.")
                    steps = data.get("steps_traced", [])
                    
                    # Render the final answers into the visual block
                    response_placeholder.markdown(generation)
                    if steps:
                        status_placeholder.caption(f"🧭 **Agent Routing Path:** { ' ➔ '.join(steps) }")
                    
                    # Append response history record to persistent state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": generation,
                        "steps": steps
                    })
                else:
                    error_msg = f"⚠️ API Server Error (Status Code: {response.status_code})"
                    response_placeholder.markdown(error_msg)
                    
            except requests.exceptions.ConnectionError:
                response_placeholder.markdown("❌ **Connection Failed.** Please make sure your FastAPI backend server is running on port 8000!")