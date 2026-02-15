import asyncio
import os
import re
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyExtract, TavilyMap

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

BATCH_SIZE = 5

tavily_map = TavilyMap()
tavily_extract = TavilyExtract()


def chunk_urls(urls: List[str], batch_size: int = BATCH_SIZE) -> List[List[str]]:
    return [urls[i : i + batch_size] for i in range(0, len(urls), batch_size)]


async def main():
    log_header("Documentation Ingestion")
    log_info("Starting the ingestion process...")

    # Step 1: Map the documentation site to discover URLs
    log_info("Mapping documentation URLs...")
    map_response = tavily_map.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 2,
        "limit": 100,
        "categories": ["Documentation"],
    })

    urls = map_response.get("results", [])
    if not urls:
        log_error("No URLs found from mapping.")
        return

    log_success(f"Mapped {len(urls)} URLs.")

    # Step 2: Chunk URLs into batches and extract content concurrently
    batches = chunk_urls(urls)
    log_info(f"Processing {len(batches)} batches of up to {BATCH_SIZE} URLs each concurrently...")

    async def extract_batch(batch: List[str], batch_num: int) -> List[Document]:
        log_info(f"Extracting batch {batch_num}/{len(batches)} ({len(batch)} URLs)...")
        try:
            extract_response = await asyncio.to_thread(
                tavily_extract.invoke, {"urls": batch}
            )

            results = extract_response.get("results", [])
            failed = extract_response.get("failed_results", [])

            if failed:
                log_warning(f"Batch {batch_num}: {len(failed)} URLs failed to extract.")

            docs = []
            for result in results:
                raw_content = result.get("raw_content", "")
                url = result.get("url", "")
                if raw_content:
                    docs.append(
                        Document(page_content=raw_content, metadata={"source": url})
                    )

            log_success(f"Batch {batch_num}: extracted {len(results)} pages.")
            return docs
        except Exception as e:
            log_error(f"Batch {batch_num} failed: {e}")
            return []

    batch_results = await asyncio.gather(
        *(extract_batch(batch, i) for i, batch in enumerate(batches, start=1))
    )

    all_documents: List[Document] = [
        doc for docs in batch_results for doc in docs
    ]

    log_success(f"Total documents extracted: {len(all_documents)}")

    if not all_documents:
        log_error("No documents extracted. Aborting ingestion.")
        return

    # Step 3: Split documents into chunks
    log_info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(all_documents)
    log_success(f"Split into {len(split_docs)} chunks.")

    # Step 4: Ingest into vector store in batches
    INGEST_BATCH_SIZE = 50
    doc_batches = [split_docs[i : i + INGEST_BATCH_SIZE] for i in range(0, len(split_docs), INGEST_BATCH_SIZE)]
    log_info(f"Ingesting {len(split_docs)} chunks into Pinecone in {len(doc_batches)} batches...")

    for i, batch in enumerate(doc_batches, start=1):
        log_info(f"Ingesting batch {i}/{len(doc_batches)} ({len(batch)} chunks)...")
        try:
            vectorstore.add_documents(batch)
            log_success(f"Batch {i}/{len(doc_batches)} ingested.")
        except Exception as e:
            log_error(f"Batch {i}/{len(doc_batches)} failed: {e}")

    log_success("Ingestion complete!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log_error(f"An error occurred: {e}")


