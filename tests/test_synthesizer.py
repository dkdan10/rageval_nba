from unittest.mock import AsyncMock, MagicMock

from rageval.demo.synthesizer import Synthesizer
from rageval.types import Document, SQLResult


async def test_deterministic_synthesizer_includes_sql_citation() -> None:
    sql = SQLResult(query="SELECT ...", rows=[{"player": "Luka Doncic", "ppg": 33.9}])

    answer = await Synthesizer().synthesize("Who led?", sql_result=sql)

    assert "[sql]" in answer
    assert "Luka Doncic" in answer


async def test_deterministic_synthesizer_includes_article_citations() -> None:
    docs = [Document(id="ctg-four-factors#0", content="The four factors explain winning.")]

    answer = await Synthesizer().synthesize("What are the four factors?", docs=docs)

    assert "[article:ctg-four-factors#0]" in answer
    assert "four factors" in answer


async def test_deterministic_synthesizer_reports_insufficient_sources() -> None:
    answer = await Synthesizer().synthesize("Unknown?")

    assert "insufficient" in answer.lower()


async def test_synthesizer_uses_injected_llm_when_available() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": "LLM answer [sql]"})

    answer = await Synthesizer(llm=llm).synthesize(
        "Who led?",
        sql_result=SQLResult(query="SELECT ...", rows=[{"player": "Luka"}]),
    )

    assert answer == "LLM answer [sql]"
    llm.complete.assert_awaited_once()


async def test_synthesizer_falls_back_when_llm_returns_empty_content() -> None:
    llm = MagicMock()
    llm.complete = AsyncMock(return_value={"content": ""})
    docs = [Document(id="doc#0", content="Evidence text.")]

    answer = await Synthesizer(llm=llm).synthesize("Question?", docs=docs)

    assert "[article:doc#0]" in answer
