# Component attribution

The application retains the local-deep-research foundation and its root MIT license.

Research control flow and portions of prompts in `runtime.py` are adapted from
[langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research).
Its MIT license is retained in this directory as `LICENSE`.

`harness.py` and the associated budget/state integration in `runtime.py` adapt
research/report allowance and unresolved-task semantics from
[jmlon/deep-research-harness](https://github.com/jmlon/deep-research-harness/tree/393d907239ee649f85dd888de715d8379e8b4a87),
revision `393d907239ee649f85dd888de715d8379e8b4a87`, specifically
`deep_research/research.py` and `deep_research/models.py`.
The upstream MIT license is retained as `HARNESS_LICENSE`.
The local implementation accounts for model calls rather than copying the
upstream token/USD accounting, and integrates the state with local source tools.

Project-specific adaptations include Collection discovery and reading,
Document-level candidate handling, mixed-source integration, local-model
execution adaptations and evaluation. These additions do not replace or remove
the upstream copyright notices.
