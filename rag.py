# rag.py
import os
import json
import re
import time
import gc
import logging
from typing import List

import chromadb
from chromadb.config import Settings  # optional when you want custom settings
from chromadb.api.models.Collection import Collection

import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# NOTE: your Groq LLM wrapper
from langchain_groq import ChatGroq

# Local helpers you already have
from scraper import scrape_with_requests, scrape_with_selenium
from preprocess import clean_html, chunk_text

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

# ---------- CONFIG ----------
# Set to True to persist to disk (can cause the readonly DB issue if process holds handles).
# Default: False (in-memory Chroma, recommended to avoid SQLite locks)
PERSIST_CHROMA = False
PERSIST_DIRECTORY = "./chroma_store"  # used only if PERSIST_CHROMA == True

# ---------- Forecast schema ----------
class ForecastItem(BaseModel):
    day: str
    max_temp: str | None = None
    min_temp: str | None = None
    condition: str | None = None
    humidity: str | None = None
    precipitation: str | None = None
    wind_speed: str | None = None
    wind_direction: str | None = None

parser = PydanticOutputParser(pydantic_object=ForecastItem)


# ---------- Vectorstore helpers ----------
def build_embeddings_model(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


import chromadb
from chromadb.config import Settings

# ---------- Vectorstore helpers ----------
def _create_chroma_client():
    """
    Create a chromadb client.
    Uses EphemeralClient for full in-memory (no persistence, no readonly issues).
    """
    try:
        client = chromadb.EphemeralClient()
        print("✅ Using EphemeralClient (pure in-memory, no file locks)")
        return client
    except Exception as e:
        print(f"⚠️ Falling back to standard in-memory client: {e}")
        return chromadb.Client(Settings(anonymized_telemetry=False))
    
def build_vectorstore(text_chunks: List[str], embeddings):
    """
    Build a vectorstore using a short-lived Chroma client.
    Stores vectordb in st.session_state as 'GLOBAL_VDB' for use by the app.
    """
    # Use a short-lived client per call to avoid long-lived file handles
    local_client = _create_chroma_client()
    collection_name = f"weather_data_{int(time.time())}"  # unique per run

    try:
        # Force remove any old collection with same name (safety)
        try:
            local_client.delete_collection(name=collection_name)
        except Exception:
            pass

        vectordb = Chroma.from_texts(
            texts=text_chunks,
            embedding=embeddings,
            client=local_client,
            collection_name=collection_name
        )

        # vectordb.persist()  # Chroma.from_texts may persist only if persist_directory provided
        st.session_state["GLOBAL_VDB"] = vectordb
        logger.info("Built vectorstore. Vectors count (if available): attempting to read.")
        # Try to get a quick count — wrap in try/except as some backends differ
        try:
            cnt = vectordb._collection.count()
            logger.info("Vector count: %s", cnt)
        except Exception:
            logger.info("Count not available for this vectorstore implementation.")
        return vectordb
    finally:
        # Remove local_client reference and force GC asap to avoid file handles lingering
        try:
            del local_client
        except Exception:
            pass
        gc.collect()
        time.sleep(0.1)


def get_global_vdb():
    return st.session_state.get("GLOBAL_VDB", None)


def delete_vector_db():
    """Completely remove the vector DB both from Streamlit and from Chroma memory."""
    # Clear any Streamlit reference
    if "GLOBAL_VDB" in st.session_state:
        del st.session_state["GLOBAL_VDB"]
        print("🧹 Deleted Streamlit session VDB reference.")

    # If you stored a client, drop it as well
    if "CHROMA_CLIENT" in st.session_state:
        client = st.session_state.pop("CHROMA_CLIENT", None)
        try:
            # Try deleting any old collections
            for col in client.list_collections():
                client.delete_collection(col.name)
            print("🧨 Deleted all Chroma collections from memory.")
        except Exception as e:
            print("⚠️ Could not delete collections cleanly:", e)

    gc.collect()
    time.sleep(0.05)
    print("✅ Full Chroma reset complete.")

# ---------- LLM helper ----------
def build_groq_llm(api_key: str, model_name: str, temperature: float = 0.1):
    return ChatGroq(api_key=api_key, model=model_name, temperature=temperature)


# ---------- Main pipeline ----------
def run_rag_extraction(
        url: str,
        groq_api_key: str,
        groq_model: str,
        use_selenium: bool = False,
        max_chunks: int = 8
):
    # 1) Scrape
    html = ""
    try:
        html = scrape_with_requests(url)
    except Exception as e:
        if use_selenium:
            html = scrape_with_selenium(url)
        else:
            try:
                html = scrape_with_selenium(url)
            except Exception as ex:
                raise RuntimeError(f"Both requests and selenium scraping failed: {e} / {ex}")

    # 2) Clean and chunk
    cleaned = clean_html(html)
    if not cleaned or len(cleaned) < 50:
        raise RuntimeError("Page cleaned to very little text — scraping may have failed.")
    chunks = chunk_text(cleaned)
    chunks = chunks[: max(1, max_chunks)]

    # 3) Remove any in-memory store and (if enabled) attempt disk cleanup
    delete_vector_db()

    # 4) Build embeddings and vectorstore
    embeddings = build_embeddings_model()
    vectordb = build_vectorstore(chunks, embeddings)

    # 5) Build retriever and LLM
    llm = build_groq_llm(api_key=groq_api_key, model_name=groq_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 6) Prepare prompt and run retrieval chain
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    user_prompt = f"""
You are given chunks of cleaned webpage text retrieved from a weather forecast page.

<context>
{{context}}
</context>

Extract all forecast items from the context and return a JSON array of forecast objects that match this schema:
Note: All temperatured are in celsius
{format_instructions}

Return ONLY valid JSON (a list of objects). If fields are unknown, use null.
Try to extract: day (e.g., "Today", "Mon", "2025-10-15"), max_temp  , min_temp , condition, humidity, precipitation, wind_speed, wind_direction.
Use concise values, e.g., "31°C", "4%", "NW 10 km/h".
Keep day names in English (e.g., "Fri 24 Oct", not localized forms).
"""
    prompt = ChatPromptTemplate.from_template(user_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    result = retrieval_chain.invoke({"input": "Extract forecast details."})
    text_output = result.get("answer", str(result))

    try:
        parsed = json.loads(text_output)
        return parsed, text_output
    except Exception:
        m = re.search(r"(\[.*\])", text_output, re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
                return parsed, text_output
            except Exception:
                pass
        return None, text_output