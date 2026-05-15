# Autonomous Research Assistant

An autonomous research agent built with **LangChain**, **Streamlit**, and **Ollama**. It combines Retrieval-Augmented Generation (RAG) with a manual ReAct reasoning loop and web scraping to provide comprehensive, source-attributed answers to research questions.

## Prerequisites

1. **Install Ollama** — download from [ollama.com](https://ollama.com/).
2. **Pull the model**:
   ```bash
   ollama pull llama3.2
   ```
3. **Start Ollama** — ensure the service is running:
   ```bash
   ollama serve
   ```

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure environment
cp .env.example .env
# Edit .env to change the model name, iteration limit, etc.
```

## Running the Application

```bash
streamlit run app.py
```

## Features

- **Local RAG** — ingest text and PDF files into a local ChromaDB vector store.
- **Autonomous Agent** — uses a ReAct loop to decide whether to query the local DB or scrape the web.
- **Web Scraping** — scrapes and ingests content from URLs when instructed.
- **Source Attribution** — shows exactly which sources influenced each answer.
- **Per-step Progress** — live status panel shows each tool call the agent makes.
- **Chat History Reset** — clear the conversation with one click from the sidebar.

## Project Structure

```
suprmentrproject/
├── app.py                  # Streamlit UI entry point
├── src/
│   ├── agent.py            # ReAct agent (LLM reasoning loop)
│   ├── ingestion.py        # Document loading & chunking
│   ├── retrieval.py        # Similarity-filtered retriever
│   ├── scraper.py          # Web scraper
│   └── vectorstore.py      # ChromaDB wrapper
├── data/
│   ├── chroma_db/          # Persisted vector store (git-ignored)
│   └── sample_docs/        # Sample documents for demo ingestion
├── .env.example            # Environment variable reference
└── requirements.txt        # Python dependencies
```
