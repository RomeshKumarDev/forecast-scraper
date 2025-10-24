# -----------------------
#  Main RAG pipeline
# -----------------------
import json, re
from typing import List

# Selenium imports (used only if user toggles it on or scraping fails)
from scraper import scrape_with_requests, scrape_with_selenium

# Preprocess imports
from preprocess import clean_html, chunk_text

# LangChain + Groq + Vector DB imports
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# NOTE: Replace this import if you use a different wrapper for Groq LLM
from langchain_groq import ChatGroq


# -----------------------
#  Forecast schema (Pydantic)
# -----------------------
class ForecastItem(BaseModel):
    day: str
    max_temp: str | None = None
    min_temp: str | None = None
    condition: str | None = None
    humidity: str | None = None
    precipitation: str | None = None
    wind_speed: str | None = None
    wind_direction: str | None = None


# Parser to enforce JSON return
parser = PydanticOutputParser(pydantic_object=ForecastItem)


def build_embeddings_model(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


# def build_vectorstore(text_chunks: List[str], embeddings) -> Chroma:
#     return Chroma.from_texts(texts=text_chunks, embedding=embeddings)
# global variable
GLOBAL_VDB = None


def build_vectorstore(text_chunks: List[str], embeddings) -> Chroma:
    global GLOBAL_VDB
    GLOBAL_VDB = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return GLOBAL_VDB


def build_groq_llm(api_key: str, model_name: str, temperature: float = 0.1):
    return ChatGroq(api_key=api_key, model=model_name, temperature=temperature)


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

    # 3) Build embeddings and vectorstore
    embeddings = build_embeddings_model()
    vectordb = build_vectorstore(chunks, embeddings)
    print(f"Vector store contains {vectordb._collection.count()} vectors.")

    # 4) Build LLM and retriever
    llm = build_groq_llm(api_key=groq_api_key, model_name=groq_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # 5) Prompt with escaped braces for parser output and context variable
    format_instructions = parser.get_format_instructions().replace("{", "{{").replace("}", "}}")

    user_prompt = f"""
You are given chunks of cleaned webpage text retrieved from a weather forecast page.

<context>
{{context}}
</context>

Extract all forecast items from the context and return a JSON array of forecast objects that match this schema:

{format_instructions}

Return ONLY valid JSON (a list of objects). If fields are unknown, use null.
Try to extract: day (e.g., "Today", "Mon", "2025-10-15"), max_temp, min_temp, condition, humidity, precipitation, wind_speed, wind_direction.
Use concise values, e.g., "31°C", "4%", "NW 10 km/h".
Keep day names in English (e.g., "Fri 24 Oct", not localized forms).
"""

    prompt = ChatPromptTemplate.from_template(user_prompt)
    print("Prompt template created successfully.")

    # 6) Create chains
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    print("Retrieval chain ready.")

    # 7) Run the chain
    result = retrieval_chain.invoke({"input": "Extract forecast details."})
    text_output = result.get("answer", str(result))

    # 8) Attempt JSON parsing
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


import shutil
import os


def delete_vector_db(vectordb=None, persist_directory=None):
    global GLOBAL_VDB
    try:
        if vectordb:
            vectordb._collection.delete(where={})
            print("🧹 In-memory vector store cleared.")
        elif GLOBAL_VDB:
            GLOBAL_VDB._collection.delete(where={})
            print("🧹 Cleared global in-memory store.")
            GLOBAL_VDB = None
        elif persist_directory and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"🗑️ Deleted Chroma persistence directory: {persist_directory}")
        else:
            print("⚠️ No vector DB found to delete.")
    except Exception as e:
        print(f"❌ Error deleting vector DB: {e}")