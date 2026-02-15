import os
import ssl

import certifi
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.agents import create_agent

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

vectorstore = PineconeVectorStore(
    index_name="angchain-docs-2025", embedding=embeddings
)

SYSTEM_PROMPT = """You are a helpful LangChain documentation assistant. You answer questions about LangChain, \
LangGraph, LangSmith, and related libraries.

Use the retrieve tool to search the documentation before answering any question. \
Always base your answers on the retrieved documentation context. If the retrieved context \
does not contain enough information to answer the question, say so clearly.

When responding:
- Provide clear, concise answers with code examples when relevant.
- Always cite your sources by including the source URLs at the end of your response.
- If the user asks about something not covered in the documentation, let them know.
"""


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Search the LangChain documentation for relevant information based on the query."""
    results = retriever.invoke(query)

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}\nContent: {doc.page_content}"
        for doc in results
    )

    return serialized, results


tools = [retrieve]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
)


if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'quit' to exit): ")
        if query.lower() in ("quit", "exit", "q"):
            break

        response = agent.invoke({"messages": [("user", query)]})

        ai_message = response["messages"][-1]
        print(f"\n{ai_message.content}")
