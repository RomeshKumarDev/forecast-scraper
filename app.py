# app.py
import streamlit as st
import os

# LangChain + Groq + Vector DB imports
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAG imports
from rag import run_rag_extraction, delete_vector_db
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()
# -----------------------
#  Config / Keys
# -----------------------
DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_GROQ_MODEL = "moonshotai/kimi-k2-instruct-0905"

# -----------------------
#  Streamlit UI
# -----------------------
st.set_page_config(page_title="RAG Weather Extractor", layout="wide")
st.title("🌦️ RAG Weather Extractor — Scrape, Clean, Embed, Retrieve, Parse")

col1, col2 = st.columns([2, 1])

with col1:
    url = st.text_input("Enter weather URL", "https://www.timeanddate.com/weather/india/delhi/ext")
    use_selenium = st.checkbox("Use Selenium if requests fails (requires chromedriver)", value=False)
    groq_api_key = st.text_input("Groq API Key", value=DEFAULT_GROQ_API_KEY, type="password")
    groq_model = st.text_input("Groq Model", value=DEFAULT_GROQ_MODEL)
    chunk_size = st.number_input("Chunk size (characters)", value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", value=200, step=50)
    max_chunks = st.number_input("Max chunks to embed", value=8, min_value=1, max_value=32, step=1)
    # Buttons side by side
    btn_col1, btn_col2 = st.columns([1, 1])

    with btn_col1:
        extract_btn = st.button("Extract forecast (RAG)", use_container_width=True)
    with btn_col2:
        delete_db_btn = st.button("🗑️ Delete Vector DB", use_container_width=True)

    if extract_btn:
        if not groq_api_key:
            st.error("Groq API key required — enter it above.")
        else:
            with st.spinner("Scraping → cleaning → embedding → retrieving..."):
                try:
                    # Update text splitter dynamically
                    def chunk_text_local(text: str):
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=int(chunk_size),
                            chunk_overlap=int(chunk_overlap),
                            separators=["\n\n", "\n", " ", ""]
                        )
                        return splitter.split_text(text)

                    globals()["chunk_text"] = chunk_text_local
                    print("gobal",globals())
                    parsed, raw_output = run_rag_extraction(
                        url=url,
                        groq_api_key=groq_api_key,
                        groq_model=groq_model,
                        use_selenium=use_selenium,
                        max_chunks=int(max_chunks)
                    )

                    if parsed is not None:
                        # --- Debug: Inspect the global Chroma vector DB ---
                        from rag import get_global_vdb
                        GLOBAL_VDB = get_global_vdb()

                        if GLOBAL_VDB:
                            st.write("### 🧠 Global Vector DB Details")
                            try:
                                count = GLOBAL_VDB._collection.count()
                                st.write(f"Number of vectors: **{count}**")

                                # Fetch a few stored documents for inspection
                                items = GLOBAL_VDB._collection.get(limit=3)
                                st.write("**Sample documents in memory:**")
                                st.json(items)

                            except Exception as e:
                                st.error(f"Error reading vector DB: {e}")
                        else:
                            st.info("No global vector DB currently loaded in memory.")

                        st.success("✅ Parsed JSON forecast (RAG)")
                        st.json(parsed)
                        st.write("### Raw LLM output (for debugging):")
                        st.code(raw_output[:10000])
                    else:
                        st.warning("LLM did not produce clean JSON. See raw output below.")
                        st.write("### Raw LLM output:")
                        st.code(raw_output[:20000])
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

    elif delete_db_btn:
        # If you persist vectors, clean that directory; otherwise, clear memory
        with st.spinner("Deleting vector database..."):
            try:
                delete_vector_db()
                st.session_state.clear()
                st.rerun()
                st.success("🧹 Vector database deleted successfully.")
            except Exception as e:
                st.error(f"Failed to delete vector DB: {e}")

with col2:
   
    st.markdown("### Quick tips")
    st.markdown("""
- Use **Selenium** for JS-heavy pages (toggle the checkbox). Make sure chromedriver is installed.
- If results are noisy, increase `max_chunks` or chunk size.
- You can swap HuggingFace embedding to a cloud embedding provider if you prefer.
- The pipeline stores vectors in-memory (Chroma) and DOES NOT persist to disk by default.
""")
    st.markdown("### Maintenance")
    st.info("Use the **🗑️ Delete Vector DB** button if you want to reset the embeddings cache.")
    st.markdown("### Dependencies (pip)")
    st.code("""
pip install streamlit bs4 requests selenium selenium-stealth chromadb langchain langchain_groq pydantic huggingface-hub sentence-transformers
    """)
