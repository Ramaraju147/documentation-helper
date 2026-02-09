import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

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
#chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="angchain-docs-2025", embedding=embeddings
)
tavily_crawl = TavilyCrawl()


async def main():
    log_info("Starting the ingestion process...")
    response = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/docs/",
        "max_depth": 2,
        "limit": 100,
    })
    print(response)
    log_success(f"Crawled {len(response)} pages successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log_error(f"An error occurred: {e}")


