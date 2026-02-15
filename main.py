import streamlit as st
from retrieval import agent

st.set_page_config(page_title="Documentation Helper", page_icon="📚")
st.header("Documentation Helper")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything about LangChain, LangGraph, or LangSmith."}
    ]

if st.sidebar.button("Clear Conversation"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything about LangChain, LangGraph, or LangSmith."}
    ]
    st.rerun()

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            chat_history = [
                (msg["role"], msg["content"])
                for msg in st.session_state["messages"]
            ]
            response = agent.invoke({"messages": chat_history})
            ai_message = response["messages"][-1].content

    st.session_state["messages"].append({"role": "assistant", "content": ai_message})
    st.chat_message("assistant").write(ai_message)
