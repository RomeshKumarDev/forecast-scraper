# -----------------------
#  Helpers: clean & chunk (preprocess.py)
# -----------------------

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from typing import List
from bs4 import BeautifulSoup
import pandas as pd
import json

def json_to_dataframe(json_data):
    """
    Takes JSON data (string or dict/list) and returns a pandas DataFrame.
    """

    # If it's a JSON string → convert to Python object
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    # If single dict → convert to list
    if isinstance(json_data, dict):
        json_data = [json_data]

    # Finally return DataFrame
    return pd.DataFrame(json_data)

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    # Remove scripts/styles/noscript
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()
    # Optionally remove ads, nav, footer by heuristics
    for sel in ["nav", "footer", ".advert", ".ads", ".cookie-banner"]:
        for t in soup.select(sel):
            t.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text)
    return text

def chunk_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)
def stack_dataframes(user_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stacks df2 below df1 and returns a combined dataframe.
    
    Both dataframes should have the same columns.
    """
    return pd.concat([weather_df,user_df ], ignore_index=True)