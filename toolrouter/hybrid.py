"""Hybrid lexical + semantic scorer.

Sometimes tool names are too short for embedding to disambiguate. A
small lexical bonus when a tool name word appears in the query goes a
long way.
"""
from __future__ import annotations

import re

import numpy as np

from .router import Router


_WORD = re.compile(r"\w+", re.UNICODE)


def hybrid_select(router: Router, query: str, k: int | None = None,
                  lexical_weight: float = 0.10) -> list:
    qv = router._normalize(router.embedder.embed([query]))[0]
    scores = router._tool_vecs @ qv

    q_words = set(_WORD.findall(query.lower()))
    for i, tool in enumerate(router._tools):
        name_words = set(_WORD.findall(tool.name.lower()))
        if q_words & name_words:
            scores[i] += lexical_weight

    if router.config.tag_boost:
        for i, t in enumerate(router._tools):
            for tag in t.tags:
                if tag in router.config.tag_boost:
                    scores[i] += router.config.tag_boost[tag]

    k = k or router.config.k
    order = scores.argsort()[::-1]
    picked: list = []
    seen: set[str] = set()
    for name in router.config.always_include:
        if name in router.registry and name not in seen:
            picked.append(router.registry.get(name))
            seen.add(name)
    for i in (int(x) for x in order):
        t = router._tools[i]
        if t.name in router.config.never_route_out or t.name in seen:
            continue
        if scores[i] < router.config.threshold:
            continue
        picked.append(t)
        seen.add(t.name)
        if len(picked) >= k:
            break
    return picked
