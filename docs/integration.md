# Integrating tool-router into an existing agent

You usually have something like this already:

```python
all_tools = [...]                       # 30+ tool definitions

resp = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[t.openai_schema() for t in all_tools],
)
```

Insert one line:

```python
relevant = router.select(messages[-1]["content"], k=6)
resp = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[t.openai_schema() for t in relevant],
)
```

That's it. The router doesn't know or care about your loop, your
state machine, or which framework you're on.

## Tips from running this in production

- **k=5–8 is the sweet spot** for most agents. Below 4, the agent can't recover from a wrong pick. Above 10, you're paying for context you don't need.
- **Always include your "submit_answer" / "finish" tool** via `always_include`. The router doesn't know that's special.
- **Reroute every turn**, not just the first. As the conversation grows the right tools shift.
- **Log the rejected tools** in development. If a useful tool keeps getting filtered out, add an `examples=[...]` line to its registry entry — it dominates the routing text.
