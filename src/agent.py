import re
import logging
from typing import Callable, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .retrieval import AdvancedRetriever
from .scraper import WebScraper
from .vectorstore import VectorStoreManager
from .ingestion import DataIngestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts — one per mode
# ---------------------------------------------------------------------------

# Mode A: KB-only.  The agent is strictly limited to the ingested documents and
# URLs.  If nothing relevant is found it must say so rather than hallucinate.
_PROMPT_KB_ONLY = """You are a professional autonomous research assistant.
You ONLY have access to the documents and websites that the user has already
ingested into the local knowledge base.  You must NOT use any outside knowledge
or make up information.

Available tools:
- query_local_knowledge_base(query): Search ingested documents and URLs.

Use the following format EXACTLY:

Thought: <your reasoning>
Action: query_local_knowledge_base
Action Input: <your search query>
Observation: <tool result>
Thought: <your reasoning>
Final Answer: <your answer>

STRICT RULES:
- You MUST call query_local_knowledge_base at least once before answering.
- If the observation contains "No relevant information found" or is clearly
  unrelated to the question, your Final Answer MUST be:
  "I could not find relevant information about this topic in the provided
  documents or links. Please ingest relevant documents or URLs first."
- Do NOT fabricate facts, statistics, or explanations not present in the
  knowledge base.
- Do NOT call scrape_website or any other tool.
- Do NOT repeat an action you already took.
- Always end with "Final Answer:" on its own line.
"""

# Mode B: KB-first, then web.  The agent queries the local KB first and only
# attempts to scrape a web URL when local results are clearly insufficient.
_PROMPT_WEB_ENABLED = """You are a professional autonomous research assistant.
Answer the user's question using the tools below.  Always prefer local
knowledge before going to the web.

Available tools:
- query_local_knowledge_base(query): Search ingested documents and URLs. Use this FIRST.
- scrape_website(url): Scrape a specific URL for additional information.
  Use ONLY when the local knowledge base returns insufficient results.
  Construct a relevant URL (e.g. a Wikipedia page, official docs, or a
  well-known resource page) based on the topic of the question.

Use the following format EXACTLY and stop after Final Answer:

Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
Observation: <tool result>
Thought: <your reasoning>
Final Answer: <your final answer to the user>

Rules:
- Always start with querying the local knowledge base.
- Only use scrape_website if local results are clearly insufficient.
- Do NOT repeat an action you already took.
- Always end with "Final Answer:" on its own line.
"""


class ResearchAgent:
    def __init__(
        self,
        llm_model: str = "llama3.2",
        vectorstore_manager: Optional[VectorStoreManager] = None,
        max_iterations: int = 8,
    ):
        """Initialise the autonomous research agent with a manual ReAct loop.

        Args:
            llm_model: Ollama model name. Defaults to ``llama3.2``.
            vectorstore_manager: Shared VectorStoreManager instance. When
                provided the agent reuses the caller's store so documents
                ingested through the UI are immediately visible to the agent.
            max_iterations: Maximum ReAct loop cycles before giving up.
        """
        try:
            self.llm = ChatOllama(model=llm_model, temperature=0)
            logger.info("LLM initialised with model '%s'.", llm_model)
        except Exception as e:
            logger.error(
                "Failed to initialise LLM '%s': %s. Ensure Ollama is running.", llm_model, e
            )
            self.llm = None

        # Reuse a shared VectorStoreManager if provided — prevents the app and
        # the agent from holding two separate ChromaDB handles that can diverge.
        self.vectorstore_manager = vectorstore_manager or VectorStoreManager()
        self.retriever = AdvancedRetriever(
            self.vectorstore_manager, similarity_threshold=0.3, k=4
        )
        self.scraper = WebScraper()
        self.ingestion = DataIngestion()
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _run_tool(self, action: str, action_input: str, web_enabled: bool) -> str:
        action = action.strip()
        action_input = action_input.strip()
        logger.debug("Tool call — action='%s', input='%s'", action, action_input[:120])

        if action == "query_local_knowledge_base":
            return self.retriever.get_context_string(action_input)

        if action == "scrape_website":
            if not web_enabled:
                # Agent tried to go to the web despite being in KB-only mode.
                # Return a polite refusal so the model can correct itself.
                return (
                    "Web access is currently disabled. "
                    "Only the local knowledge base is available. "
                    "If you have no relevant local results, state that in your Final Answer."
                )
            doc = self.scraper.scrape_url(action_input)
            if doc:
                chunks = self.ingestion.process_and_chunk([doc])
                self.vectorstore_manager.add_documents(chunks)
                return (
                    f"Successfully scraped {action_input} and added {len(chunks)} chunks. "
                    "You can now query the knowledge base for information about it."
                )
            return f"Failed to scrape {action_input}."

        return f"Unknown tool: {action}"

    # ------------------------------------------------------------------
    # KB-only: direct retrieve → answer (no ReAct format needed)
    # ------------------------------------------------------------------

    def _answer_from_kb(
        self,
        query: str,
        step_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Retrieve relevant context from the KB then ask the LLM to answer
        using ONLY that context.

        Avoids the full ReAct format entirely, which small models (tinyllama,
        llama3.2) routinely fail to follow, producing prompt fragments or
        hallucinated tool calls instead of real answers.
        """
        if step_callback:
            step_callback("🔍 **Searching knowledge base …**")

        context = self.retriever.get_context_string(query)

        if "No relevant information found" in context:
            logger.info("KB-only: no relevant documents found for query.")
            return (
                "I could not find relevant information about this topic "
                "in the provided documents or links. "
                "Please ingest relevant documents or URLs first."
            )

        if step_callback:
            preview = context[:120] + ("…" if len(context) > 120 else "")
            step_callback(f"📋 **Retrieved context:** {preview}")

        # Plain Q&A prompt — no tool-call syntax, works with any model size.
        prompt = (
            "You are a research assistant. "
            "Answer the question using ONLY the context provided below. "
            "Do not use any outside knowledge. "
            "If the context does not contain enough information to answer, "
            "say so clearly.\n\n"
            f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        logger.debug("KB-only direct answer prompt sent to LLM.")
        response = self.llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        logger.info("KB-only direct answer produced (%d chars).", len(answer))
        return answer

    # ------------------------------------------------------------------
    # ReAct loop (web-enabled mode)
    # ------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        step_callback: Optional[Callable[[str], None]] = None,
        web_search_enabled: bool = False,
    ) -> str:
        """Answer *query* using the appropriate mode.

        Args:
            query: The user's research question.
            step_callback: Optional callable invoked with a status string at
                each step (useful for streaming progress to a UI).
            web_search_enabled: When False (default) the agent retrieves from
                the local KB and answers strictly from that content — no ReAct
                loop, no internet.  When True the full ReAct loop runs and the
                agent may scrape web URLs when local results are insufficient.
        """
        if not self.llm:
            return (
                "Agent is not initialised. "
                "Please ensure Ollama is running and the model is pulled."
            )

        # KB-only: bypass the ReAct loop entirely for reliability with small models.
        if not web_search_enabled:
            return self._answer_from_kb(query, step_callback)

        # Web-enabled: full ReAct loop.

        system_prompt = _PROMPT_WEB_ENABLED

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        for iteration in range(self.max_iterations):
            logger.debug("ReAct iteration %d / %d", iteration + 1, self.max_iterations)
            response = self.llm.invoke(messages)
            text = response.content

            # ── Final Answer ──────────────────────────────────────────────
            # Stop capture before the next Thought/Action block so trailing
            # boilerplate is not included in the answer shown to the user.
            fa_match = re.search(
                r"Final Answer\s*:\s*(.*?)(?=\n\s*(?:Thought|Action)\s*:|\Z)",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if fa_match:
                answer = fa_match.group(1).strip()
                logger.info("Final Answer produced after %d iteration(s).", iteration + 1)
                return answer

            # ── Action / Action Input ─────────────────────────────────────
            action_match = re.search(r"Action\s*:\s*(.+?)(?=\n|$)", text, re.IGNORECASE)
            input_match = re.search(
                r"Action Input\s*:\s*(.*?)(?=\n\s*(?:Thought|Action|Observation)\s*:|\Z)",
                text,
                re.DOTALL | re.IGNORECASE,
            )

            if action_match and input_match:
                action = action_match.group(1).strip()
                action_input = input_match.group(1).strip()

                if step_callback:
                    step_callback(f"🔧 **Tool:** `{action}`  |  **Input:** `{action_input[:80]}`")

                observation = self._run_tool(action, action_input, web_enabled=web_search_enabled)
                logger.debug("Observation (first 200 chars): %s", observation[:200])

                if step_callback:
                    preview = observation[:120] + ("…" if len(observation) > 120 else "")
                    step_callback(f"📋 **Observation:** {preview}")

                # Inject the tool result with a clear label so the model does
                # not confuse it with user speech.
                messages.append(AIMessage(content=text))
                messages.append(
                    HumanMessage(
                        content=(
                            "[Tool Result]\n"
                            f"Action: {action}\n"
                            f"Observation: {observation}\n"
                            "[End Tool Result]\n\n"
                            "Continue your reasoning."
                        )
                    )
                )
            else:
                # Model gave a plain answer without invoking any tool.
                if not web_search_enabled:
                    # KB-only mode: the model must NOT answer from training data.
                    # Force-run the KB query ourselves to validate.
                    kb_result = self.retriever.get_context_string(query)
                    if "No relevant information found" in kb_result:
                        logger.info(
                            "KB-only: model skipped tool and KB is empty — returning not-found."
                        )
                        return (
                            "I could not find relevant information about this topic "
                            "in the provided documents or links. "
                            "Please ingest relevant documents or URLs first."
                        )
                    # KB has content — inject it and re-prompt so the model
                    # answers from the actual knowledge base, not its weights.
                    logger.info(
                        "KB-only: model skipped tool but KB has content — injecting and re-prompting."
                    )
                    if step_callback:
                        step_callback("🔧 **Tool:** `query_local_knowledge_base` (auto-forced)")
                    messages.append(AIMessage(content=text))
                    messages.append(
                        HumanMessage(
                            content=(
                                "[Tool Result]\n"
                                "Action: query_local_knowledge_base\n"
                                f"Observation: {kb_result}\n"
                                "[End Tool Result]\n\n"
                                "Based ONLY on the above knowledge base results, "
                                "provide your Final Answer. "
                                "Do NOT use any information from outside those results."
                            )
                        )
                    )
                    continue  # re-enter loop with KB context injected

                logger.info("Direct answer (no tool call) on iteration %d.", iteration + 1)
                return text.strip()

        logger.warning("ReAct loop exhausted after %d iterations.", self.max_iterations)
        return (
            f"I was unable to reach a conclusive answer within {self.max_iterations} steps. "
            "Try rephrasing your question or ingesting more relevant documents."
        )