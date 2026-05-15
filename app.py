import logging
import os
import tempfile
import warnings

# Suppress noisy transformers deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.utils.generic")
warnings.filterwarnings("ignore", message="Accessing `__path__` from*")

import streamlit as st
from dotenv import load_dotenv

from src.agent import ResearchAgent
from src.ingestion import DataIngestion
from src.retrieval import AdvancedRetriever
from src.scraper import WebScraper
from src.vectorstore import VectorStoreManager

# ---------------------------------------------------------------------------
# Environment & logging setup
# ---------------------------------------------------------------------------

load_dotenv()  # Load variables from .env if present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Autonomous Research Assistant",
    layout="wide",
    page_icon="🕵️",
)

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------


@st.cache_resource
def init_vectorstore() -> VectorStoreManager:
    return VectorStoreManager()


@st.cache_resource
def init_ingestion() -> DataIngestion:
    return DataIngestion()


@st.cache_resource
def init_scraper() -> WebScraper:
    return WebScraper()


@st.cache_resource
def init_agent(_vectorstore: VectorStoreManager) -> ResearchAgent:
    """Pass the shared VectorStoreManager so the agent and UI see the same DB.

    The leading underscore tells Streamlit not to attempt hashing this arg.
    """
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    max_iter = int(os.getenv("AGENT_MAX_ITERATIONS", "8"))
    return ResearchAgent(
        llm_model=model,
        vectorstore_manager=_vectorstore,
        max_iterations=max_iter,
    )


# Initialise all components; surface friendly errors instead of crashing.
try:
    vectorstore = init_vectorstore()
    ingestion = init_ingestion()
    scraper = init_scraper()
    agent = init_agent(vectorstore)
    # Retriever used independently for source-attribution display in the UI.
    retriever = AdvancedRetriever(vectorstore, similarity_threshold=0.3, k=4)
    _init_error: str | None = None
except Exception as _exc:
    _init_error = str(_exc)
    logger.exception("Fatal error during component initialisation.")
    vectorstore = ingestion = scraper = agent = retriever = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("🕵️ Autonomous Research Assistant")

# Initialise toggle state before sidebar renders so it's always available.
if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

if _init_error:
    st.error(
        f"**Initialisation failed:** {_init_error}\n\n"
        "Please ensure Ollama is running (`ollama serve`) and all dependencies are installed."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — control panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Control Panel")

    # --- 1. Ingest sample docs ---
    st.subheader("1. Ingest Sample Documents")
    if st.button("📂 Ingest from data/sample_docs"):
        with st.spinner("Ingesting data/sample_docs …"):
            chunks = ingestion.ingest_data_folder()  # uses absolute default path
            if chunks:
                vectorstore.add_documents(chunks)
                st.success(f"Added {len(chunks)} chunks to the knowledge base.")
            else:
                st.warning("No documents found in data/sample_docs.")

    st.divider()

    # --- 2. Upload files ---
    st.subheader("2. Upload Files")
    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Files are chunked and added to the local vector database.",
    )

    if st.button("📥 Ingest Uploaded Files") and uploaded_files:
        all_chunks = []
        with st.spinner("Processing uploads …"):
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                # Write to temp file, close it, THEN hand path to loader.
                # This avoids Windows file-locking issues where PyMuPDF can
                # hold an open handle that prevents os.unlink inside finally.
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                # File is closed here — safe to pass to any loader.

                try:
                    if suffix.lower() == ".pdf":
                        docs = ingestion.load_pdf(tmp_path)
                    else:
                        docs = ingestion.load_single_text_file(tmp_path)

                    # Restore original filename as source metadata.
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name

                    chunks = ingestion.process_and_chunk(docs)
                    all_chunks.extend(chunks)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    logger.exception("Failed to process upload '%s'.", uploaded_file.name)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except PermissionError:
                        # On Windows a loader may still hold the handle briefly.
                        # The OS will clean it up on next reboot; log and continue.
                        logger.warning("Could not delete temp file '%s'; skipping.", tmp_path)
                    except OSError as e:
                        logger.warning("Temp file cleanup failed: %s", e)

        if all_chunks:
            vectorstore.add_documents(all_chunks)
            st.success(f"Added {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")
        else:
            st.warning("No content could be extracted from the uploaded files.")

    st.divider()

    # --- 3. Scrape Websites ---
    st.subheader("3. Scrape Websites")
    st.caption("Enter one URL per line.")
    urls_input = st.text_area(
        "URLs to scrape",
        placeholder="https://example.com/article\nhttps://another-site.com/page",
        height=120,
        label_visibility="collapsed",
    )
    if st.button("🌐 Scrape & Ingest URLs"):
        # Parse and validate URLs from the text area
        raw_lines = urls_input.strip().splitlines()
        valid_urls = [u.strip() for u in raw_lines if u.strip().startswith("http")]
        invalid_lines = [u.strip() for u in raw_lines if u.strip() and not u.strip().startswith("http")]

        if invalid_lines:
            st.warning(
                f"Skipped {len(invalid_lines)} invalid line(s) — URLs must start with http:// or https://:\n"
                + "\n".join(f"- `{u}`" for u in invalid_lines)
            )

        if not valid_urls:
            st.error("No valid URLs found. Please enter at least one URL starting with http:// or https://")
        else:
            all_chunks = []
            progress = st.progress(0, text="Starting …")
            status_placeholder = st.empty()

            results = {"succeeded": [], "failed": []}
            for idx, url in enumerate(valid_urls):
                progress.progress(
                    (idx) / len(valid_urls),
                    text=f"Scraping {idx + 1}/{len(valid_urls)}: {url[:60]}…",
                )
                doc = scraper.scrape_url(url)
                if doc:
                    chunks = ingestion.process_and_chunk([doc])
                    all_chunks.extend(chunks)
                    results["succeeded"].append((url, doc.metadata.get("title", url), len(chunks)))
                else:
                    results["failed"].append(url)

            progress.progress(1.0, text="Done!")

            if all_chunks:
                vectorstore.add_documents(all_chunks)

            # ── Summary ──────────────────────────────────────────────
            if results["succeeded"]:
                st.success(
                    f"✅ {len(results['succeeded'])} URL(s) ingested — "
                    f"{len(all_chunks)} chunks added to knowledge base."
                )
                with st.expander("📄 View details", expanded=False):
                    for url, title, chunks in results["succeeded"]:
                        st.markdown(f"**{title}**")
                        st.caption(f"{url}")
                        st.markdown(f"`{chunks} chunks`")
                        st.divider()
            if results["failed"]:
                st.error(f"❌ {len(results['failed'])} URL(s) failed to scrape.")
                with st.expander("View failed URLs", expanded=False):
                    for u in results["failed"]:
                        st.caption(u)

    st.divider()

    # --- 4. Web Search Mode ---
    st.subheader("4. Web Search Mode")
    web_search_enabled = st.toggle(
        "🌍 Enable web search",
        value=st.session_state.web_search_enabled,
        help=(
            "OFF — Agent answers only from ingested documents and URLs.\n\n"
            "ON — Agent queries local knowledge base first, then scrapes "
            "the web if local results are insufficient."
        ),
    )
    st.session_state.web_search_enabled = web_search_enabled

    if web_search_enabled:
        st.info("🌍 Web search **enabled** — agent will query local docs first, then go online if needed.")
    else:
        st.info("🔒 Web search **disabled** — agent only uses ingested documents and links.")

    st.divider()

    # --- 5. Manage chat history ---
    st.subheader("5. Chat History")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header("💬 Chat with the Assistant")

# Mode badge shown below the header.
if st.session_state.web_search_enabled:
    st.caption("🌍 **Mode:** KB-first + Web fallback — local documents are searched first; web is used only when needed.")
else:
    st.caption("🔒 **Mode:** Knowledge Base only — answers come strictly from your ingested documents and links.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Re-render source attribution stored alongside past assistant messages.
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources used", expanded=False):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['source']}** &nbsp; "
                        f"<span style='color:grey;font-size:0.85em;'>score: {src['score']:.2f}</span>",
                        unsafe_allow_html=True,
                    )

# New message
if prompt := st.chat_input("Ask a research question …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Use st.status to stream per-step agent progress to the user.
        with st.status("🤔 Agent is thinking …", expanded=True) as status:
            step_messages: list[str] = []

            def _on_step(msg: str) -> None:
                step_messages.append(msg)
                status.write(msg)

            response = agent.run_query(
                prompt,
                step_callback=_on_step,
                web_search_enabled=st.session_state.web_search_enabled,
            )
            status.update(label="✅ Done!", state="complete", expanded=False)

        st.markdown(response)

        # --- Source attribution ---
        source_docs = retriever.invoke(prompt)
        sources = []
        seen_keys: set = set()
        for doc in source_docs:
            src_name = doc.metadata.get("source", "Unknown")
            score = doc.metadata.get("similarity_score", 0.0)
            key = (src_name, round(score, 2))
            if key not in seen_keys:
                seen_keys.add(key)
                sources.append({"source": src_name, "score": score})

        if sources:
            with st.expander("📚 Sources used", expanded=False):
                for src in sources:
                    st.markdown(
                        f"- **{src['source']}** &nbsp; "
                        f"<span style='color:grey;font-size:0.85em;'>score: {src['score']:.2f}</span>",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )