# QuantV2 Instructions

This project is an event-driven equity quant research system.

The first goal is not live trading. The first goal is a clean, reproducible research pipeline that prevents lookahead bias.

Rules:

- Every feature must be known at or before the decision timestamp.
- Every label must be generated strictly after the decision timestamp.
- Never use future data when building features.
- Do not add live trading or brokerage integration unless explicitly requested.
- Prefer simple baselines before complex models.
- Add tests for label timing, feature timing, and no-lookahead behavior.
- Do not commit raw market data, API keys, credentials, or `.env` files.
