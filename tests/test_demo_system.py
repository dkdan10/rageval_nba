import pytest

from rageval.demo.system import HybridRAGSystem
from rageval.types import Document, QuestionType, SQLResult


class FakeRouter:
    def __init__(self, route: QuestionType) -> None:
        self.route = route
        self.questions: list[str] = []

    async def classify(self, question: str) -> QuestionType:
        self.questions.append(question)
        return self.route


class FakeSQLAgent:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    async def generate_and_execute(self, question: str) -> SQLResult:
        self.calls.append(question)
        if self.fail:
            raise RuntimeError("sql down")
        return SQLResult(query="SELECT 1", rows=[{"answer": 1}])


class FakeRAGAgent:
    def __init__(
        self,
        *,
        fail: bool = False,
        docs: list[Document] | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> None:
        self.fail = fail
        self.docs = docs
        self.last_retrieval_diagnostics = diagnostics or {}
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, k: int = 5) -> list[Document]:
        self.calls.append((question, k))
        if self.fail:
            raise RuntimeError("rag down")
        if self.docs is not None:
            return self.docs
        return [Document(id="doc#0", content="Article evidence.")]


class FakeSynthesizer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, SQLResult | None, list[Document] | None]] = []

    async def synthesize(
        self,
        question: str,
        sql_result: SQLResult | None = None,
        docs: list[Document] | None = None,
    ) -> str:
        self.calls.append((question, sql_result, docs))
        if self.fail:
            raise RuntimeError("synth down")
        parts: list[str] = []
        if sql_result is not None:
            parts.append("[sql]")
        if docs:
            parts.append("[article:doc#0]")
        return " ".join(parts) or "no evidence"


class FakeCostLLM:
    def __init__(self) -> None:
        self.total_cost_usd = 0.0


class CostRouter:
    def __init__(self, llm: FakeCostLLM, route: QuestionType) -> None:
        self._llm = llm
        self.route = route

    async def classify(self, _question: str) -> QuestionType:
        self._llm.total_cost_usd += 0.01
        return self.route


class CostSQLAgent:
    def __init__(self, llm: FakeCostLLM) -> None:
        self._llm = llm

    async def generate_and_execute(self, _question: str) -> SQLResult:
        self._llm.total_cost_usd += 0.02
        return SQLResult(query="SELECT 1", rows=[{"answer": 1}])


class CostSynthesizer:
    def __init__(self, llm: FakeCostLLM) -> None:
        self._llm = llm

    async def synthesize(
        self,
        _question: str,
        sql_result: SQLResult | None = None,
        docs: list[Document] | None = None,
    ) -> str:
        self._llm.total_cost_usd += 0.03
        return "answer"


async def test_factual_route_calls_sql_only() -> None:
    sql = FakeSQLAgent()
    rag = FakeRAGAgent()
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.FACTUAL),
        sql_agent=sql,
        rag_agent=rag,
        synthesizer=FakeSynthesizer(),
    )

    response = await system.answer("Who led PPG?")

    assert response.routing_decision is QuestionType.FACTUAL
    assert response.sql_result is not None
    assert response.retrieved_docs == []
    assert len(sql.calls) == 1
    assert rag.calls == []
    assert response.refused is False
    assert response.latency_ms is not None


async def test_analytical_route_calls_rag_only() -> None:
    sql = FakeSQLAgent()
    rag = FakeRAGAgent()
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.ANALYTICAL),
        sql_agent=sql,
        rag_agent=rag,
        synthesizer=FakeSynthesizer(),
    )

    response = await system.answer("What are four factors?")

    assert response.routing_decision is QuestionType.ANALYTICAL
    assert response.sql_result is None
    assert len(response.retrieved_docs) == 1
    assert sql.calls == []
    assert len(rag.calls) == 1


async def test_hybrid_route_calls_sql_and_rag() -> None:
    sql = FakeSQLAgent()
    rag = FakeRAGAgent()
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.HYBRID),
        sql_agent=sql,
        rag_agent=rag,
        synthesizer=FakeSynthesizer(),
        top_k=3,
    )

    response = await system.answer("Stats and analysis?")

    assert response.routing_decision is QuestionType.HYBRID
    assert response.sql_result is not None
    assert len(response.retrieved_docs) == 1
    assert len(sql.calls) == 1
    assert rag.calls == [("Stats and analysis?", 3)]


async def test_rag_retrieval_diagnostics_survive_empty_doc_fallback() -> None:
    rag = FakeRAGAgent(
        docs=[],
        diagnostics={
            "requested_mode": "vector",
            "retrieval_mode": "lexical_fallback",
            "fallback_reason": "no_vector_results",
            "fallback_doc_count": 0,
        },
    )
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.ANALYTICAL),
        sql_agent=FakeSQLAgent(),
        rag_agent=rag,
        synthesizer=FakeSynthesizer(),
    )

    response = await system.answer("What are four factors?")

    assert response.retrieved_docs == []
    assert response.metadata["retrieval"]["retrieval_mode"] == "lexical_fallback"
    assert response.metadata["retrieval"]["fallback_reason"] == "no_vector_results"


async def test_unanswerable_route_refuses_without_paths() -> None:
    sql = FakeSQLAgent()
    rag = FakeRAGAgent()
    synth = FakeSynthesizer()
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.UNANSWERABLE),
        sql_agent=sql,
        rag_agent=rag,
        synthesizer=synth,
    )

    response = await system.answer("Who will win MVP in 2028?")

    assert response.refused is True
    assert response.routing_decision is QuestionType.UNANSWERABLE
    assert "cannot answer" in response.answer.lower()
    assert sql.calls == []
    assert rag.calls == []
    assert synth.calls == []


async def test_response_cost_is_per_case_delta_not_cumulative() -> None:
    llm = FakeCostLLM()
    system = HybridRAGSystem(
        router=CostRouter(llm, QuestionType.FACTUAL),
        sql_agent=CostSQLAgent(llm),
        rag_agent=FakeRAGAgent(),
        synthesizer=CostSynthesizer(llm),
    )

    first = await system.answer("Who led PPG?")
    second = await system.answer("Who led PPG again?")

    assert first.cost_usd == 0.06
    assert second.cost_usd == pytest.approx(0.06)
    assert llm.total_cost_usd == pytest.approx(0.12)


async def test_path_errors_do_not_crash_response() -> None:
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.HYBRID),
        sql_agent=FakeSQLAgent(fail=True),
        rag_agent=FakeRAGAgent(fail=True),
        synthesizer=FakeSynthesizer(),
    )

    response = await system.answer("Stats and analysis?")

    assert response.routing_decision is QuestionType.HYBRID
    assert response.sql_result is not None
    assert response.sql_result.error is not None
    assert "sql down" in response.answer
    assert "rag down" in response.answer


async def test_synthesizer_error_does_not_crash_response() -> None:
    system = HybridRAGSystem(
        router=FakeRouter(QuestionType.FACTUAL),
        sql_agent=FakeSQLAgent(),
        rag_agent=FakeRAGAgent(),
        synthesizer=FakeSynthesizer(fail=True),
    )

    response = await system.answer("Who led PPG?")

    assert "could not synthesize" in response.answer.lower()
    assert "synth down" in response.answer
