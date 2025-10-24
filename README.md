# RAG Weather Extractor

A Streamlit app that scrapes weather forecast pages, cleans and chunks the HTML, stores relevant chunks in a Chroma vector store, and uses a retrieval-augmented-generation (RAG) pipeline with a Groq LLM to extract structured JSON forecasts. Supports Selenium for JS-heavy pages and can persist or keep vector data in-memory.

---

## Table of Contents

- What this project does
- Features
- Folder structure
- Requirements
- Installation (local)
- Configuration & secrets
- How it works — step by step
- Usage (Streamlit UI)
- Persistence & Viewing Vector DB
- Deployment to Streamlit Community Cloud
- Notes on Selenium & headless scraping
- Troubleshooting
- Contributing
- License

---

## What this project does

This app turns messy HTML weather pages into tidy, machine-readable JSON forecasts. Given a URL, the pipeline:

1. Scrapes the page (via requests; optional Selenium fallback for JS-rendered pages).
2. Cleans the HTML and removes irrelevant elements (scripts, ads, nav, etc.).
3. Breaks the cleaned text into chunks for focused retrieval.
4. Embeds the chunks (HuggingFace embeddings by default) and stores them in a Chroma vector store (in-memory or persisted to disk).
5. Uses a Groq LLM via LangChain retrieval to extract and return a JSON list of forecast objects with fields:  
   `day, max_temp, min_temp, condition, humidity, precipitation, wind_speed, wind_direction`
6. Presents the parsed JSON and raw LLM output in the Streamlit UI for debugging.

---

## Features

- Requests-first scraping with optional Selenium fallback
- HTML cleaning and chunking (RecursiveCharacterTextSplitter)
- In-memory or persistent Chroma vector store
- RAG pipeline via LangChain + Groq LLM
- Pydantic output parsing for strict JSON schema
- Streamlit UI for inputs, settings, and results
- Option for user to choose whether to persist vector data

---

## Folder Structure (suggested)

```
Weather-Extractor/
├─ app.py
├─ preprocess.py
├─ scraper.py
├─ rag.py
├─ requirements.txt
├─ .gitignore
├─ README.md
└─ (optional) weather_vectors/   # if persisting Chroma
```

---

## Requirements

List all Python dependencies in `requirements.txt`. Example:

```
streamlit
bs4
requests
selenium
selenium-stealth
chromadb
langchain
langchain_groq
pydantic
huggingface-hub
sentence-transformers
```

If you are not using Selenium (recommended for Streamlit Community Cloud), you can remove `selenium` and `selenium-stealth`.

---

## Installation (local)

1. **Clone the repo:**
   ```
   git clone https://github.com/<your-username>/Weather-Extractor.git
   cd Weather-Extractor
   ```
2. **Create & activate a virtualenv:**
   ```
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   # .\venv\Scripts\activate  # Windows PowerShell
   ```
3. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **(Optional)** If you plan to use Selenium locally, make sure Chrome and chromedriver are installed and chromedriver is in PATH.

---

## Configuration & Secrets

**Never commit secrets to the repository.**

Read the Groq API key from an environment variable or from Streamlit secrets.

- **macOS / Linux:**
  ```
  export DEFAULT_GROQ_API_KEY="your_groq_api_key"
  ```
- **Windows PowerShell:**
  ```
  $env:DEFAULT_GROQ_API_KEY="your_groq_api_key"
  ```

In Streamlit Community Cloud, add the secret via the app dashboard:  
Settings → Secrets

In `app.py`, use:
```python
import os
DEFAULT_GROQ_API_KEY = os.getenv("DEFAULT_GROQ_API_KEY")
```

---

## How it works — step by step

1. User inputs a URL and Groq API key (or the app reads it from environment).
2. Scrape stage: `scrape_with_requests()` fetches the HTML. If that fails or if user opted in, `scrape_with_selenium()` loads the page with headless Chrome for JS-rendered content.
3. Cleaning stage: `clean_html()` removes `<script>`, `<style>`, `<iframe>`, ads, nav, footer, and normalizes whitespace.
4. Chunking stage: `RecursiveCharacterTextSplitter` splits the text into chunks (size and overlap configurable).
5. Embedding and vector store: Each chunk is embedded using a HuggingFace model. Chroma stores these vectors — either in-memory (default) or persisted to a directory.
6. Retrieval & generation: A LangChain RetrievalQA chain fetches the top-k chunks and the Groq LLM (via `langchain_groq.ChatGroq`) is prompted to produce strict JSON conforming to the ForecastItem pydantic schema.
7. Output: The app tries to `json.loads()` the model response; if that fails, it searches for a JSON substring; otherwise, the raw output is shown for debugging.

---

## Usage (Streamlit UI)

Open the app in browser (local):

```
streamlit run app.py
```

UI elements:

- Enter weather URL
- Use Selenium if requests fails (toggle)
- Groq API Key (or set as environment variable)
- Groq Model (default: moonshotai/kimi-k2-instruct-0905)
- Chunk size / overlap
- Max chunks (cost-control)
- Extract forecast (RAG): Runs the pipeline and displays results

---

## Persistence & Viewing Vector DB

- The vector store is in-memory by default (Chroma with no `persist_directory`).
- To persist vectors to disk, enable the “Store vector data locally” option (or set `persist_directory` when creating Chroma).
- Persisted data will appear in a folder like `./weather_vectors/` containing `chroma.sqlite3` and index files.
- To inspect stored chunks programmatically:
  ```python
  docs = vectordb.get()
  # or with LangChain API
  retriever = vectordb.as_retriever()
  ```
- For Streamlit UI, you can add a viewer for stored chunk texts and metadata (display only with user consent).

---

## Deployment to Streamlit Community Cloud

1. Push code to GitHub. Make sure `requirements.txt` is present and no secrets are committed.
2. If secrets were ever committed, purge them from Git history (BFG or git filter-repo) and force-push a cleaned history.
3. On Streamlit Cloud: New app → connect your GitHub repo → select branch + `app.py`.
4. Add secrets in the app dashboard (Settings → Secrets), e.g.:
   ```
   DEFAULT_GROQ_API_KEY="your_groq_api_key"
   ```
5. Deploy. Streamlit Cloud will run `pip install -r requirements.txt` automatically.

**Note:** Streamlit Community Cloud does not provide Chrome/Chromedriver by default — avoid Selenium or use a remote scraping solution for production.

---

## Notes on Selenium & Headless Scraping

- Selenium requires a system Chrome binary + matching chromedriver binary. On local machines you can install both.
- Streamlit Community Cloud typically does not include Chrome/chromedriver — Selenium may fail there. Prefer requests + BeautifulSoup for deployment to Streamlit Cloud.
- Alternative options for JS-heavy pages:
  - Use a hosted headless browser service (e.g., Browserless, Playwright Cloud, ScraperAPI).
  - Use the site’s API if available (many weather sites provide an API or JSON endpoint).

---

## Troubleshooting

- **ModuleNotFoundError:**  
  Ensure `requirements.txt` contains the missing package. Run `pip freeze > requirements.txt` locally to capture all installed packages.

- **Push blocked by GitHub secret scanning:**  
  If your commit history contains secrets, GitHub will block pushes. Use BFG or git filter-repo to purge secrets from history and force-push the cleaned branch.

- **Selenium errors on Streamlit Cloud:**  
  Remove Selenium usage or move scraping to a background worker / external service.

- **LLM returns non-JSON:**  
  Increase `max_chunks`, refine the prompt, or add stricter output parsing (Pydantic + format instructions). Display raw LLM output for debugging.

---
