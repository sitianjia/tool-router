# ToolRouter — pick K tools before they hit the LLM

A tiny routing layer that sits between your user query and your agent. It takes a free-text request, scores it against your tool catalogue, and hands the agent only the few tools that actually matter.

## Why?

Once you have more than a dozen tools, dumping all of them into every LLM call gets expensive fast: 1-2k input tokens of schema, every turn, every user. The model also gets dumber — the more options it sees, the more often it picks a wrong one or invents a bad call.

For 80% of queries, the right `k=5` tools are obvious from a cosine similarity score. Use the embedding model for that step. Save the big model's context for the actual reasoning.

## Install & Run

```bash
pip install -r requirements.txt
```

```python
from toolrouter import Tool, ToolRegistry, Router
from toolrouter.embedders import OpenAIEmbedder

registry = ToolRegistry([
    Tool("weather_now",     "Get current weather for a city",
         examples=["What's the weather in Beijing?"], tags=["realtime"]),
    Tool("currency_convert","Convert between currencies",
         examples=["USD to CNY"], tags=["finance"]),
    Tool("sql_query",       "Run a SQL query against the data warehouse",
         tags=["data"]),
    # ... and so on
])

router = Router(registry, OpenAIEmbedder())
picks = router.select("how much is 50 USD in yuan?", k=3)
for t in picks:
    print(t.name, t.description)
# currency_convert  Convert between currencies
# ...
```

You then pass `picks` (or their schemas) to your agent loop. The model only sees the tools that the router thinks are relevant.

## Configuration

```python
from toolrouter.router import RouterConfig

router.config = RouterConfig(
    k=5,
    threshold=0.15,                       # drop very low-confidence picks (cosine; -1..1)
    tag_boost={"realtime": 0.05},         # nudge time-sensitive tools up
    always_include=["finish_task"],       # control-flow tools always shown
    never_route_out=["secret_tool"],      # never expose without manual opt-in
)
```

| Field | Default | What it does |
|-------|---------|--------------|
| `k` | `5` | max tools to return |
| `threshold` | `0.0` | drop tools with cosine score below this |
| `tag_boost` | `{}` | additive score bonus per tag |
| `always_include` | `[]` | tools to prepend before scoring |
| `never_route_out` | `[]` | tools to hide from auto-routing |

## Examples

**Caching**: tool embeddings are computed once at construction. Persist with `router.save(path)` and reload with `Router.load(path, ...)` — useful when starting many worker processes from the same catalogue.

**Custom embedder**: implement `BaseEmbedder.embed(texts) -> np.ndarray`. Cohere, BGE, jina-embeddings-v3 all work fine.

**Score debugging**:

```python
scores = router.score("how much is 50 USD in yuan?")
for name, s in sorted(scores.items(), key=lambda x: -x[1])[:5]:
    print(f"{s:+.3f}  {name}")
```

If the right tool isn't in the top of `score()`, you usually want to add an `examples=[...]` line to that tool — it dominates the routing text.

## What this isn't

- It is **not** an agent loop. Drop the selected tools into whatever loop you already have.
- It is **not** a re-ranker. Top-k by cosine is the whole algorithm. If you need cross-encoder rerank, layer it yourself.
- It is **not** trained. Bring your own embedder. Anything that produces vectors works.

## License

MIT.
