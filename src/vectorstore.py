import hashlib
import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Absolute path anchored to this file so the DB is always found regardless of
# the working directory at launch time.
_DEFAULT_PERSIST_DIR = str(
    Path(__file__).resolve().parent.parent / "data" / "chroma_db"
)


class VectorStoreManager:
    def __init__(
        self,
        persist_directory: str = _DEFAULT_PERSIST_DIR,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model '%s'.", embedding_model_name)
        # HuggingFace embeddings run fully locally — no API key needed.
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        self.vector_store = Chroma(
            collection_name="research_agent_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        logger.info("ChromaDB ready at '%s'.", self.persist_directory)

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_id(page_content: str) -> str:
        """Derive a stable ID from chunk content using SHA-256.

        ChromaDB treats IDs as primary keys: a chunk with the same ID is
        *updated* rather than duplicated, preventing silent inflation when a
        user clicks "Ingest" multiple times.

        SHA-256 is used instead of MD5 to eliminate the (admittedly small)
        risk of hash collisions silently overwriting unrelated chunks.
        """
        return hashlib.sha256(page_content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_documents(self, chunks: list) -> list[str]:
        """Add document chunks to the vector store, skipping exact duplicates."""
        if not chunks:
            logger.warning("add_documents called with an empty chunk list.")
            return []

        # Stable, content-derived IDs — re-ingesting the same file is a no-op.
        # Deduplicate within the incoming batch to avoid DuplicateIDError.
        seen: dict[str, object] = {}
        for chunk in chunks:
            cid = self._chunk_id(chunk.page_content)
            if cid not in seen:
                seen[cid] = chunk
        unique_ids = list(seen.keys())
        unique_chunks = list(seen.values())

        batch_size = 100
        for i in range(0, len(unique_chunks), batch_size):
            batch_chunks = unique_chunks[i : i + batch_size]
            batch_ids = unique_ids[i : i + batch_size]
            self.vector_store.add_documents(documents=batch_chunks, ids=batch_ids)
            logger.info(
                "Batch %d added (%d chunks).", i // batch_size + 1, len(batch_chunks)
            )

        logger.info("Done — %d unique chunk(s) stored in ChromaDB.", len(unique_chunks))
        return unique_ids

    def get_retriever(self, k: int = 4):
        """Return a LangChain retriever interface for the vector store."""
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = VectorStoreManager()
    print("Vector store initialised successfully.")