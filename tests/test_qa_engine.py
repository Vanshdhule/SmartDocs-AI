# tests/test_qa_engine.py
"""
Unit tests for QAEngine — now using LangChain ChatOpenAI.
All LangChain LLM calls are mocked — no API key or credits required.

Mock target: 'backend.qa_engine._build_langchain_llm'
This replaces the LLM factory used by QAEngine.__init__ and switch_model(),
so every chain built inside QAEngine uses our fake LLM.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "text":            "The Eiffel Tower was completed in 1889 in Paris, France.",
        "source":          "paris_guide.pdf",
        "page":            4,
        "chunk_index":     0,
        "relevance_score": 0.92,
    },
    {
        "text":            "Gustave Eiffel designed the tower for the 1889 World Fair.",
        "source":          "paris_guide.pdf",
        "page":            5,
        "chunk_index":     1,
        "relevance_score": 0.88,
    },
]

SAMPLE_ANSWER = (
    "The Eiffel Tower was completed in 1889 "
    "[Source: paris_guide.pdf, Page: 4] and was designed by Gustave Eiffel "
    "[Source: paris_guide.pdf, Page: 5]."
)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _make_mock_llm(answer: str = SAMPLE_ANSWER):
    """
    Build a mock that behaves like a LangChain ChatOpenAI instance.
    Supports both .invoke() (blocking) and .stream() (token generator).
    The LCEL | operator builds a chain — we mock the whole chain's invoke/stream
    by patching _build_langchain_llm to return an object whose __or__ and
    __ror__ produce a runnable that returns our fake answer.
    """
    mock_llm = MagicMock()

    # Make the | operator (LCEL chain building) return a chain that works
    def make_chain(*args, **kwargs):
        chain = MagicMock()
        chain.invoke.return_value = answer
        chain.stream.return_value = iter(answer.split())
        # Support further chaining (prompt | llm | parser)
        chain.__or__ = make_chain
        chain.__ror__ = make_chain
        return chain

    mock_llm.__or__  = make_chain
    mock_llm.__ror__ = make_chain
    mock_llm.invoke.return_value  = MagicMock(content=answer)
    mock_llm.stream.return_value  = iter([MagicMock(content=t) for t in answer.split()])
    return mock_llm


def _make_mock_memory():
    """Mock ConversationBufferWindowMemory."""
    mem = MagicMock()
    mem.load_memory_variables.return_value = {"history": []}
    return mem


@pytest.fixture
def engine():
    """
    QAEngine with all LangChain LLM calls mocked.
    Patches _build_langchain_llm so no network calls are made.
    """
    mock_llm = _make_mock_llm()

    with patch("backend.qa_engine._build_langchain_llm", return_value=mock_llm), \
         patch("backend.qa_engine.ConversationBufferWindowMemory",
               return_value=_make_mock_memory()):

        from backend.qa_engine import QAEngine
        eng = QAEngine()

        # Patch the pre-built chains directly so invoke/stream work correctly
        eng._chain        = MagicMock()
        eng._chain_stream = MagicMock()
        eng._chain.invoke.return_value        = SAMPLE_ANSWER
        eng._chain_stream.stream.return_value = iter(SAMPLE_ANSWER.split())

        yield eng


# ── Tests: generate_answer ────────────────────────────────────────────────────

class TestGenerateAnswer:
    def test_returns_answer_string(self, engine):
        result = engine.generate_answer("When was it built?", SAMPLE_CHUNKS)
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_result_has_required_keys(self, engine):
        result = engine.generate_answer("question", SAMPLE_CHUNKS)
        assert "answer"               in result
        assert "citations"            in result
        assert "quality_check_passed" in result
        assert "model_used"           in result

    def test_conversation_stored_after_answer(self, engine):
        engine.generate_answer("first question?", SAMPLE_CHUNKS)
        assert len(engine.conversation_history) == 1


# ── Tests: conversation history ───────────────────────────────────────────────

class TestConversationHistory:
    def test_store_and_retrieve(self, engine):
        engine.store_conversation_turn("Q1", "A1")
        msgs = engine.get_history_messages()
        assert any(m["content"] == "Q1" for m in msgs)
        assert any(m["content"] == "A1" for m in msgs)

    def test_get_history_returns_list_when_empty(self, engine):
        assert engine.get_history_messages() == []

    def test_history_capped_at_three_turns(self, engine):
        for i in range(6):
            engine.store_conversation_turn(f"Q{i}", f"A{i}")
        assert len(engine.conversation_history) == 3

    def test_clear_memory(self, engine):
        engine.store_conversation_turn("Q", "A")
        engine.clear_memory()
        assert engine.conversation_history == []


# ── Tests: citations ──────────────────────────────────────────────────────────

class TestParseCitations:
    def test_extracts_structured_citations(self, engine):
        cites = engine.parse_citations(SAMPLE_ANSWER)
        assert len(cites) >= 1
        assert cites[0]["file"] == "paris_guide.pdf"
        assert cites[0]["page"] == 4

    def test_deduplicates_citations(self, engine):
        duped = SAMPLE_ANSWER + " [Source: paris_guide.pdf, Page: 4]"
        cites = engine.parse_citations(duped)
        pairs = [(c["file"], c["page"]) for c in cites]
        assert len(pairs) == len(set(pairs))

    def test_no_citations_returns_empty(self, engine):
        assert engine.parse_citations("No sources here.") == []


# ── Tests: quality validation ─────────────────────────────────────────────────

class TestValidateAnswerQuality:
    def test_good_answer_passes(self, engine):
        assert engine.validate_answer_quality(SAMPLE_ANSWER, SAMPLE_CHUNKS) is True

    def test_empty_answer_fails(self, engine):
        assert engine.validate_answer_quality("", SAMPLE_CHUNKS) is False

    def test_too_short_fails(self, engine):
        assert engine.validate_answer_quality("Yes.", SAMPLE_CHUNKS) is False

    def test_no_citation_fails(self, engine):
        long_no_cite = "The tower was built in 1889 and stands tall in Paris centre."
        assert engine.validate_answer_quality(long_no_cite, SAMPLE_CHUNKS) is False

    def test_blanket_refusal_fails(self, engine):
        refusal = (
            "The provided documents do not contain this information about "
            "Eiffel Tower height dimensions [Source: paris_guide.pdf, Page: 4]."
        )
        assert engine.validate_answer_quality(refusal, SAMPLE_CHUNKS) is False


# ── Tests: context formatting ─────────────────────────────────────────────────

class TestFormatContext:
    def test_includes_source_and_page(self, engine):
        ctx = engine.format_context(SAMPLE_CHUNKS)
        assert "paris_guide.pdf" in ctx
        assert "4" in ctx

    def test_all_chunks_included(self, engine):
        ctx = engine.format_context(SAMPLE_CHUNKS)
        for chunk in SAMPLE_CHUNKS:
            assert chunk["text"][:20] in ctx


# ── Tests: streaming ──────────────────────────────────────────────────────────

class TestStreamAnswer:
    def test_stream_yields_strings(self, engine):
        engine._chain_stream.stream.return_value = iter(
            ["The ", "tower ", "was ", "built."]
        )
        tokens = list(engine.stream_answer("When?", SAMPLE_CHUNKS))
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_stream_concatenates_to_full_answer(self, engine):
        words = ["Eiffel ", "Tower ", "1889 ", "[Source: paris_guide.pdf, Page: 4]"]
        engine._chain_stream.stream.return_value = iter(words)
        full = "".join(engine.stream_answer("When?", SAMPLE_CHUNKS))
        assert "Eiffel" in full


# ── Tests: split_reasoning ────────────────────────────────────────────────────

class TestSplitReasoning:
    def test_no_think_block_returns_none_and_original(self, engine):
        reasoning, answer = engine.split_reasoning("Plain answer here.")
        assert reasoning is None
        assert answer == "Plain answer here."

    def test_think_block_is_extracted(self, engine):
        raw = "<think>step 1\nstep 2</think>Final answer."
        reasoning, answer = engine.split_reasoning(raw)
        assert "step 1" in reasoning
        assert answer == "Final answer."

    def test_answer_without_think_is_clean(self, engine):
        raw = "<think>internal</think>Clean response."
        _, answer = engine.split_reasoning(raw)
        assert "<think>" not in answer


# ── Tests: switch_model ───────────────────────────────────────────────────────

class TestSwitchModel:
    def test_switch_updates_model_name(self, engine):
        original = engine.model
        new_model = "nvidia/llama-3.1-nemotron-70b-instruct"
        with patch("backend.qa_engine._build_langchain_llm",
                   return_value=_make_mock_llm()):
            engine.switch_model(new_model)
        assert engine.model == new_model

    def test_switch_to_same_model_is_noop(self, engine):
        original = engine.model
        with patch("backend.qa_engine._build_langchain_llm",
                   return_value=_make_mock_llm()) as mock_build:
            engine.switch_model(original)
            mock_build.assert_not_called()   # no rebuild needed