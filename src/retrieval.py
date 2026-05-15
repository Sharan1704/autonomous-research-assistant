import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    """Custom retriever with similarity-score filtering.

    Score-direction auto-detection
    --------------------------------
    ``similarity_search_with_relevance_scores`` should return *similarity*
    scores in [0, 1] where 1 = most similar.  However, depending on how the
    ChromaDB collection was created it can return raw *distance* scores instead
    (lower = more similar).

    On the first query this class inspects the returned scores and sets
    ``_score_is_distance`` automatically so the threshold filter is always
    applied in the correct direction.  You can also force the direction via the
    ``score_is_distance`` constructor argument.
    """

    def __init__(
        self,
        vectorstore_manager,
        similarity_threshold: float = 0.3,
        k: int = 4,
        score_is_distance: Optional[bool] = None,
    ):
        self.vectorstore_manager = vectorstore_manager
        self.similarity_threshold = similarity_threshold
        self.k = k
        # None = auto-detect on first query; True/False = caller-forced
        self._score_is_distance: Optional[bool] = score_is_distance
        self._direction_probed: bool = score_is_distance is not None

    # ------------------------------------------------------------------
    # Score-direction probe
    # ------------------------------------------------------------------

    def _probe_direction(self, results: list) -> bool:
        """Return True if scores appear to be distances (best = lowest).

        Requires strong evidence before flipping direction:
        - At least 3 results (can't tell from 1–2 data points)
        - Minimum score is very close to 0 (< 0.02)
        - More than half the scores are above 0.5
          (genuine similarity scores cluster in [0.1, 0.9]; distance scores
          after normalisation tend to spread the same way but the BEST match
          is near 0, not near 1)

        This conservative threshold prevents a single low-scoring irrelevant
        document from accidentally triggering the direction flip and letting
        all weak matches through.
        """
        if len(results) < 3:
            return False
        scores = [score for _, score in results]
        min_score = min(scores)
        high_count = sum(1 for s in scores if s > 0.5)
        return min_score < 0.02 and high_count > len(scores) / 2

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def _retrieve_raw(self, query: str) -> list:
        return self.vectorstore_manager.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k
        )

    def invoke(self, query: str) -> List[Document]:
        """Retrieve and filter documents by similarity score.

        This is the modern LangChain-compatible entry point (replaces the
        deprecated ``get_relevant_documents``).
        """
        results = self._retrieve_raw(query)

        # Auto-detect score direction on the very first call.
        if not self._direction_probed and results:
            self._score_is_distance = self._probe_direction(results)
            self._direction_probed = True
            direction = "distance" if self._score_is_distance else "similarity"
            logger.info("Score direction auto-detected: %s.", direction)

        filtered_docs: List[Document] = []
        for doc, score in results:
            logger.debug(
                "score=%.4f | source=%s", score, doc.metadata.get("source", "?")
            )
            # Invert the comparison when scores are distances (lower = better).
            passes = (
                score <= self.similarity_threshold
                if self._score_is_distance
                else score >= self.similarity_threshold
            )
            if passes:
                doc.metadata["similarity_score"] = score
                filtered_docs.append(doc)

        return filtered_docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Deprecated — use ``invoke()`` instead.  Kept for back-compat."""
        logger.warning(
            "get_relevant_documents() is deprecated; call invoke() instead."
        )
        return self.invoke(query)

    # ------------------------------------------------------------------
    # Context string helper
    # ------------------------------------------------------------------

    def get_context_string(self, query: str) -> str:
        """Return a formatted string of retrieved documents for LLM context."""
        docs = self.invoke(query)

        if not docs:
            return "No relevant information found in the local knowledge base."

        context = ""
        for doc in docs:
            source = doc.metadata.get("source", "Unknown Source")
            score = doc.metadata.get("similarity_score", 0.0)
            context += f"\n[Source: {source} | Score: {score:.2f}]\n{doc.page_content}\n"

        return context