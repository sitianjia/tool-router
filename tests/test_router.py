import numpy as np

from toolrouter import Tool, ToolRegistry, Router
from toolrouter.embedders import BagOfWordsEmbedder
from toolrouter.router import RouterConfig


def _make() -> Router:
    reg = ToolRegistry([
        Tool("weather_now", "Get current weather for a city",
             examples=["What's the weather in Beijing?"], tags=["realtime"]),
        Tool("currency_convert", "Convert between currencies",
             examples=["USD to CNY"], tags=["finance"]),
        Tool("sql_query", "Run a SQL query against the warehouse",
             examples=["select count(*) from orders"], tags=["data"]),
        Tool("send_email", "Send an email to a recipient", tags=["comms"]),
        Tool("file_search", "Search the internal docs index", tags=["docs"]),
    ])
    return Router(reg, BagOfWordsEmbedder(dim=512))


def test_relevant_tool_first():
    r = _make()
    top = [t.name for t in r.select("how is the weather in shanghai today?", k=1)]
    assert top == ["weather_now"]


def test_k_caps_results():
    r = _make()
    assert len(r.select("anything", k=3)) <= 3


def test_always_include():
    r = _make()
    r.config.always_include = ["send_email"]
    names = [t.name for t in r.select("compute 1+1", k=2)]
    assert names[0] == "send_email"


def test_score_returns_all():
    r = _make()
    s = r.score("anything")
    assert set(s.keys()) == {"weather_now", "currency_convert",
                             "sql_query", "send_email", "file_search"}


def test_save_load(tmp_path):
    r = _make()
    r.save(tmp_path / "snap")
    r2 = Router.load(tmp_path / "snap", r.registry, r.embedder)
    np.testing.assert_array_almost_equal(r._tool_vecs, r2._tool_vecs)
