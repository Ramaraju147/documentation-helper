import streamlit as st
from langchain_core.messages import ToolMessage

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
    if msg.get("sources"):
        with st.chat_message("assistant"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.write(source)

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

            sources = []
            for msg in response["messages"]:
                if isinstance(msg, ToolMessage) and msg.artifact:
                    for doc in msg.artifact:
                        source = doc.metadata.get("source", "")
                        if source and source not in sources:
                            sources.append(source)

        st.write(ai_message)
        if sources:
            with st.expander("Sources"):
                for source in sources:
                    st.write(source)

    st.session_state["messages"].append(
        {"role": "assistant", "content": ai_message, "sources": sources}
    )
