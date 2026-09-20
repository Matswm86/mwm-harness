# COMPAT: measured model behaviour on the shared endpoint

Written by `spike/spike_toolcalls.py` on 2026-09-20 09:52 UTC. Endpoint: `http://localhost:11434/v1`.
Every value below is measured, none is assumed. Raw chunks: `spike/out/`.

| model | 2 parallel tool calls | arguments valid JSON | sent reasoning text | finish_reason | prompt / completion tokens | cached tokens on repeat | note |
|---|---|---|---|---|---|---|---|
| qwen3-vl:4b-instruct | yes | yes | no | tool_calls | 236 / 44 | 235 | 5.2s for 2 requests |
| qwen3-vl:4b | yes | yes | yes | tool_calls | 238 / 213 | 237 | 11.3s for 2 requests |
| gemma4:e4b | yes | yes | no | tool_calls | 138 / 33 | 133 | 23.5s for 2 requests |
| qwen3-vl:8b | yes | yes | yes | tool_calls | 238 / 263 | 237 | 65.5s for 2 requests |
