# Changelog

## [0.2.0] - 2025-11
- Cross-encoder rerank pass (`toolrouter.rerank`)
- Batched OpenAI embedding requests
- Bumped minimum openai>=1.30

## [0.1.0] - 2025-04
- First public version
- `Tool` / `ToolRegistry` core
- `Router` with cosine top-k, tag boost, must-include rules
- Embedders: OpenAI-compatible API, hashing BoW, BGE / sentence-transformers
- YAML registry loader
- Hybrid lexical+semantic selector
- Persistence (`router.save` / `Router.load`)
