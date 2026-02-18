from toolrouter import Tool, ToolRegistry, Router
from toolrouter.embedders import BagOfWordsEmbedder


def test_add_then_route():
    reg = ToolRegistry([Tool("a", "alpha")])
    r = Router(reg, BagOfWordsEmbedder(dim=256))
    r.add_tool(Tool("b", "beta about the weather"))
    names = [t.name for t in r.select("how is the weather", k=1)]
    assert names == ["b"]


def test_remove():
    reg = ToolRegistry([Tool("a", "alpha"), Tool("b", "beta")])
    r = Router(reg, BagOfWordsEmbedder(dim=256))
    r.remove_tool("b")
    assert [t.name for t in r.select("anything", k=5)] == ["a"]
