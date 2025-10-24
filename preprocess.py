# -----------------------
#  Helpers: clean & chunk
# -----------------------

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from typing import List
from bs4 import BeautifulSoup

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
