# app.py
import streamlit as st
import os
from preprocess import json_to_dataframe, stack_dataframes
import pandas as pd

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
st.title(" RAG Weather Extractor")

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
        extract_btn = st.button("Scrape Weather Data", use_container_width=True)
    with btn_col2:
        delete_db_btn = st.button("Delete Vector DB", use_container_width=True)

    st.write("Session State", st.session_state.get("weather_df"))
    # Extraction logic
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
                    print("gobal", globals())
                    parsed, raw_output = run_rag_extraction(
                        url=url,
                        groq_api_key=groq_api_key,
                        groq_model=groq_model,
                        use_selenium=use_selenium,
                        max_chunks=int(max_chunks)
                    )

                    if parsed is not None:
                        # # --- Debug: Inspect the global Chroma vector DB ---
                        # from rag import get_global_vdb
                        # GLOBAL_VDB = get_global_vdb()

                        # if GLOBAL_VDB:
                        #     st.write("### 🧠 Global Vector DB Details")
                        #     try:
                        #         count = GLOBAL_VDB._collection.count()
                        #         st.write(f"Number of vectors: **{count}**")

                        #         # Fetch a few stored documents for inspection
                        #         items = GLOBAL_VDB._collection.get(limit=3)
                        #         st.write("**Sample documents in memory:**")
                        #         st.json(items)

                        #     except Exception as e:
                        #         st.error(f"Error reading vector DB: {e}")
                        # else:
                        #     st.info("No global vector DB currently loaded in memory.")

                        st.success("Parsed json forecast data successfully")
                        st.json(parsed)
                        weather_df = json_to_dataframe(parsed)
                        st.session_state["weather_df"] = weather_df
                        st.markdown("## Forecast DataFrame")
                        st.dataframe(weather_df)

                        # File UPload Code    

                        # Read the file
                        st.session_state["user_df"] = pd.read_csv("data.csv")

                        





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

                #  CHANGE: CLEAR session after saving
                if st.session_state.get("weather_df"):
                    del st.session_state["weather_df"]
                elif st.session_state.get("user_df"):
                    del st.session_state["user_df"]

                st.info("Session cleared — scraped data removed from memory.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to delete vector DB: {e}")
    # Show Add Button ONLY if df already has data
    df = st.session_state.get("weather_df")

    if df is not None and not df.empty:
        if st.button("Add Scraped Data to Existing DataFrame"):
            # safe to use df
    
            # 🔥 CHANGE: Use weather_df from session_state, NOT from local variable
            weather_df = st.session_state.get("weather_df")
            user_df = st.session_state.get("user_df")
    
            if weather_df is None:
                st.error("No scraped data found in session. Please scrape again.")
            else:
                # Merge dataframes
                df = stack_dataframes(weather_df, user_df)
    
                # Save back to CSV
                df.to_csv("data.csv", index=False)
    
                st.success("Data added successfully to CSV!")
                st.write("## Updated Combined DataFrame")
                st.dataframe(df)

with col2:
    st.markdown("### Maintenance")
    st.info("Use the **Delete Vector DB** button if you want to reset the embeddings cache.")