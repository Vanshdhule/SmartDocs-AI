# backend/qa_engine.py
"""
Context-aware Q&A Engine — powered by LangChain + NVIDIA NIM

LangChain components used
──────────────────────────
• ChatOpenAI          — LLM wrapper (supports NVIDIA NIM base_url)
• ChatPromptTemplate  — structured prompt with system + history + context
• MessagesPlaceholder — injects conversation memory into the prompt
• ConversationBufferWindowMemory — stores last K turns
• LCEL chain (prompt | llm | StrOutputParser) — composable pipeline
• Streaming           — native LangChain streaming via .stream()

Supports two modes
───────────────────
• generate_answer()  — blocking, returns full dict
• stream_answer()    — generator, yields tokens one by one
"""

import re
import os
import logging
from typing import List, Dict, Iterator

import tiktoken
from dotenv import load_dotenv

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

try:
    from config import (
        NVIDIA_BASE_URL, NVIDIA_CHAT_API_KEY,
        CHAT_MODEL, QA_MAX_CONTEXT_TOKENS,
        QA_TEMPERATURE, CONVERSATION_HISTORY_TURNS,
    )
except ImportError:
    NVIDIA_BASE_URL            = "https://integrate.api.nvidia.com/v1"
    NVIDIA_CHAT_API_KEY        = None
    CHAT_MODEL                 = "meta/llama-3.3-70b-instruct"
    QA_MAX_CONTEXT_TOKENS      = 32_000
    QA_TEMPERATURE             = 0.2
    CONVERSATION_HISTORY_TURNS = 3

load_dotenv()
logger = logging.getLogger("smartdocs.qa_engine")

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = (
    "You are an expert AI assistant for document-based question answering.\n"
    "RULES:\n"
    "1. Answer using ONLY the CONTEXT provided below — do not use outside knowledge.\n"
    "2. Give COMPREHENSIVE, DETAILED answers. Extract and explain actual content,\n"
    "   events, characters, findings, and conclusions from the context.\n"
    "3. For questions spanning multiple documents: synthesize information from ALL\n"
    "   provided sources and clearly compare or combine what each document says.\n"
    "4. For summary requests: synthesize key ideas and give a structured response.\n"
    "5. Cite facts inline using EXACTLY this format: [Source: filename.pdf, Page: N]\n"
    "6. IMPORTANT — Only say 'The provided documents do not contain this information'\n"
    "   if the context is COMPLETELY silent on the topic. If the context has ANY\n"
    "   relevant passages — even partial or indirect ones — use them to answer.\n"
    "   Do NOT refuse if you can provide even a partial answer.\n"
    "7. Never fabricate information. Be factual, clear, and thorough.\n\n"
    "CONTEXT:\n{context}"
)


def _build_langchain_llm(model: str, temperature: float = QA_TEMPERATURE,
                          streaming: bool = False) -> ChatOpenAI:
    """
    Factory that creates a LangChain ChatOpenAI pointed at NVIDIA NIM.
    Uses NVIDIA_CHAT_API_KEY from config / .env.
    """
    api_key = NVIDIA_CHAT_API_KEY or os.getenv("NVIDIA_CHAT_API_KEY")
    if not api_key:
        raise ValueError("❌ NVIDIA_CHAT_API_KEY not found. Set it in .env.")
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=NVIDIA_BASE_URL,
        temperature=temperature,
        streaming=streaming,
        max_retries=3,
    )


class QAEngine:
    """
    RAG Q&A engine built on LangChain LCEL.

    Chain architecture:
        ChatPromptTemplate  (system + memory + user question)
            │
        ChatOpenAI          (NVIDIA NIM — Llama 3.3 or DeepSeek R1)
            │
        StrOutputParser     (extracts text from AIMessage)
    """

    _STOP_WORDS = {
        "a","an","the","and","or","but","in","on","at","to","for","of","with",
        "is","was","are","were","be","been","being","have","has","had","do",
        "does","did","will","would","could","should","may","might","shall",
        "this","that","these","those","it","its","i","you","he","she","we",
        "they","what","which","who","how","when","where","why","not","no",
        "so","if","as","by","from","into","through","about","than","then",
    }

    def __init__(
        self,
        model:       str   = CHAT_MODEL,
        max_tokens:  int   = QA_MAX_CONTEXT_TOKENS,
        temperature: float = QA_TEMPERATURE,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

        # tiktoken for token counting (model-agnostic fallback to gpt-3.5)
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # ── LangChain components ───────────────────────────────────────────────
        # Non-streaming LLM (for generate_answer)
        self._llm = _build_langchain_llm(model, temperature, streaming=False)

        # Streaming LLM (for stream_answer)
        self._llm_stream = _build_langchain_llm(model, temperature, streaming=True)

        # Prompt template with conversation memory slot
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        # Output parser — extracts plain text from AIMessage
        self._parser = StrOutputParser()

        # LCEL chains
        self._chain        = self._prompt | self._llm        | self._parser
        self._chain_stream = self._prompt | self._llm_stream | self._parser

        # Conversation memory: last CONVERSATION_HISTORY_TURNS turns
        self._memory = ConversationBufferWindowMemory(
            k=CONVERSATION_HISTORY_TURNS,
            return_messages=True,
            memory_key="history",
        )

        # Legacy list kept for backwards compat
        self.conversation_history: List[Dict[str, str]] = []

    # ── Model switching ────────────────────────────────────────────────────────

    def switch_model(self, new_model: str) -> None:
        """
        Hot-swap the underlying LLM at runtime (e.g. Llama ↔ DeepSeek R1).
        Resets conversation memory to avoid cross-model context bleed.
        """
        if new_model == self.model:
            return
        logger.info(f"Switching QA model: {self.model} → {new_model}")
        self.model        = new_model
        self._llm         = _build_langchain_llm(new_model, self.temperature, False)
        self._llm_stream  = _build_langchain_llm(new_model, self.temperature, True)
        self._chain       = self._prompt | self._llm        | self._parser
        self._chain_stream= self._prompt | self._llm_stream | self._parser
        self._memory.clear()
        self.conversation_history = []
        logger.info(f"Model switched to {new_model}")

    # ── Context & prompt helpers ───────────────────────────────────────────────

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """Format retrieved chunks into a numbered, attributed context block."""
        return "\n".join(
            f"[{i}] Source: {c['source']}, Page: {c['page']}\n{c['text']}\n"
            for i, c in enumerate(retrieved_chunks, 1)
        )

    def _get_history_messages(self):
        """Load conversation history from LangChain memory."""
        return self._memory.load_memory_variables({}).get("history", [])

    def _build_chain_input(self, question: str, context: str) -> Dict:
        """Assemble the input dict for the LCEL chain."""
        # Token-limit check: trim context if needed
        ctx_tokens = len(self.encoding.encode(context))
        if ctx_tokens > self.max_tokens - 512:
            # Trim to fit — keep the start (most relevant chunks)
            allowed = self.max_tokens - 512
            tokens  = self.encoding.encode(context)[:allowed]
            context = self.encoding.decode(tokens)
            logger.warning(f"Context trimmed to {allowed} tokens")
        return {
            "context":  context,
            "history":  self._get_history_messages(),
            "question": question,
        }

    # ── DeepSeek R1 reasoning trace helper ────────────────────────────────────

    @staticmethod
    def split_reasoning(answer: str):
        """
        DeepSeek R1 wraps its chain-of-thought in <think>…</think> tags.
        Returns (reasoning_trace, final_answer).
        If no think block exists, reasoning_trace is None.
        """
        match = re.search(r"<think>(.*?)</think>\s*", answer, re.DOTALL)
        if match:
            return match.group(1).strip(), answer[match.end():].strip()
        return None, answer

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_answer(
        self, question: str, retrieved_chunks: List[Dict]
    ) -> Dict:
        """Blocking call — returns full answer dict."""
        context  = self.format_context(retrieved_chunks)
        inp      = self._build_chain_input(question, context)

        # Invoke LangChain LCEL chain
        raw_answer = self._chain.invoke(inp)

        # For DeepSeek R1: separate reasoning from final answer
        reasoning, answer = self.split_reasoning(raw_answer)

        # Store in LangChain memory + legacy list
        self._memory.save_context(
            {"input": question}, {"output": answer}
        )
        self.store_conversation_turn(question, answer)

        return {
            "answer":               answer,
            "reasoning_trace":      reasoning,     # None for non-reasoning models
            "citations":            self.parse_citations(answer),
            "quality_check_passed": self.validate_answer_quality(
                answer, retrieved_chunks
            ),
            "model_used":           self.model,
        }

    def stream_answer(
        self, question: str, retrieved_chunks: List[Dict]
    ) -> Iterator[str]:
        """
        Streaming call — yields text tokens one by one via LangChain .stream().
        For DeepSeek R1, <think> tokens are yielded but can be filtered by
        the caller using split_reasoning() on the accumulated string.
        """
        context = self.format_context(retrieved_chunks)
        inp     = self._build_chain_input(question, context)
        yield from self._chain_stream.stream(inp)

    def summarize_document(self, chunks: List[Dict], doc_name: str) -> str:
        """
        Generate a one-paragraph summary of a document from its first N chunks.
        Called by the ingestion pipeline after indexing.
        Uses the non-streaming LLM.
        """
        if not chunks:
            return ""
        sample_text = "\n\n".join(
            f"[Page {c.get('page_number', c.get('page', '?'))}] "
            f"{c.get('text', '')[:400]}"
            for c in chunks
        )
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a document analyst. Write a concise 2-3 sentence summary "
             "of the following document excerpts. Be factual and professional. "
             "Do not use bullet points."),
            ("human",
             f"Document: {doc_name}\n\nExcerpts:\n{sample_text}\n\nSummary:"),
        ])
        chain  = summary_prompt | self._llm | self._parser
        try:
            result = chain.invoke({})
            _, clean = self.split_reasoning(result)
            return clean.strip()
        except Exception as e:
            logger.warning(f"Summarisation failed for '{doc_name}': {e}")
            return ""

    # ── Conversation memory ────────────────────────────────────────────────────

    def store_conversation_turn(self, question: str, answer: str) -> None:
        """Append to legacy conversation list (kept for backwards compat)."""
        self.conversation_history.append({"question": question, "answer": answer})
        self.conversation_history = (
            self.conversation_history[-CONVERSATION_HISTORY_TURNS:]
        )

    def get_history_messages(self) -> List[Dict[str, str]]:
        """Return history as OpenAI-style message dicts (for compatibility)."""
        msgs = []
        for turn in self.conversation_history[-CONVERSATION_HISTORY_TURNS:]:
            msgs.append({"role": "user",      "content": turn["question"]})
            msgs.append({"role": "assistant",  "content": turn["answer"]})
        return msgs

    def add_conversation_history(self, question=None, answer=None):
        """Deprecated shim — kept for backwards compatibility."""
        if question and answer:
            self.store_conversation_turn(question, answer)
            return
        return self.get_history_messages()

    def clear_memory(self) -> None:
        """Clear both LangChain memory and legacy list."""
        self._memory.clear()
        self.conversation_history = []

    # ── Citation helpers ───────────────────────────────────────────────────────

    def extract_citations(self, answer: str) -> List[str]:
        return re.findall(r"\[(.*?, page \d+)\]", answer)

    def parse_citations(self, answer: str) -> List[Dict]:
        pattern = r"\[Source:\s*([^,]+),\s*Page:\s*(\d+)\]"
        seen, citations = set(), []
        for fname, page_str in re.findall(pattern, answer):
            key = (fname.strip(), page_str.strip())
            if key not in seen:
                seen.add(key)
                citations.append({"file": fname.strip(), "page": int(page_str)})
        return citations

    # ── Quality validation ─────────────────────────────────────────────────────

    def validate_answer_quality(
        self, answer: str, retrieved_chunks: List[Dict]
    ) -> bool:
        """
        Four-signal quality check:
        1. Length ≥ 10 words
        2. Not a blanket refusal when context exists
        3. At least one [Source: ..., Page: N] citation
        4. ≥ 3 meaningful (non-stop-word) tokens overlap with context
        """
        # Strip reasoning trace before checking
        _, clean_answer = self.split_reasoning(answer)
        if not clean_answer or len(clean_answer.split()) < 10:
            return False
        if "do not contain this information" in clean_answer.lower() and retrieved_chunks:
            return False
        if not re.search(r"\[Source:\s*[^,]+,\s*Page:\s*\d+\]", clean_answer):
            return False
        context_text = " ".join(
            c.get("text", "") for c in retrieved_chunks
        ).lower()
        meaningful = {
            w.strip(".,;:!?\"'()").lower()
            for w in clean_answer.split()
            if w.lower() not in self._STOP_WORDS and len(w) > 3
        }
        return sum(1 for w in meaningful if w in context_text) >= 3