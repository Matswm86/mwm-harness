# COMPAT: measured model behaviour on the shared endpoint

Written by `spike/spike_toolcalls.py` on 2026-09-20 09:52 UTC. Endpoint: `http://localhost:11434/v1`.
Every value below is measured, none is assumed. Raw chunks: `spike/out/`.

| model | 2 parallel tool calls | arguments valid JSON | sent reasoning text | finish_reason | prompt / completion tokens | cached tokens on repeat | note |
|---|---|---|---|---|---|---|---|
| qwen3-vl:4b-instruct | yes | yes | no | tool_calls | 236 / 44 | 235 | 5.2s for 2 requests |
| qwen3-vl:4b | yes | yes | yes | tool_calls | 238 / 213 | 237 | 11.3s for 2 requests |
| gemma4:e4b | yes | yes | no | tool_calls | 138 / 33 | 133 | 23.5s for 2 requests |
| qwen3-vl:8b | yes | yes | yes | tool_calls | 238 / 263 | 237 | 65.5s for 2 requests |
| ornith-1.5-35b-16k | yes | yes | yes | tool_calls | 357 / 85 | 0 | 114.6s for 2 requests (2026-10-04, includes a model load) |

## 2026-10-04: Ornith-1.5-35B-A3B vs qwen3-vl-4b-instruct-16k on two bug-fix tasks

Ornith = official `ornith-ai/Ornith-1.5-35B-A3B-GGUF` Q4_K_M (21.7 GB) as the 16k
variant `ornith-1.5-35b-16k`. Ollama 0.34.0 put the expert weights in system RAM
and 41 shared layers (~2 GB) on the 6 GB RTX 3060 laptop GPU. Warm speed on a
347-token answer: 18.8 tokens/s generated. The first spike attempt timed out at
the script's 120 s limit during the cold load.

Each run: headless `mwm -p`, `--no-mcp --no-hooks --mode bypassPermissions`,
fresh copy of the task, 900 s cap. Task a = `parse.py` crashes on "1,234.50";
task b = `median()` wrong for even-length lists, 2 failing pytest tests, tests
must stay untouched. Pass = the program/test result, checked by script.

| model | task a pass | task b pass | requests per pass | seconds per pass |
|---|---|---|---|---|
| qwen3-vl-4b-instruct-16k | 2/3 | 1/3 | 5-6 | 209-348 |
| ornith-1.5-35b-16k | 3/3 | 2/3 | 4-6 | 465-806 |

- 4B failures: a-2 replaced "," with "." (still crashes), then the repeat brake
  ended the turn; b-2 still working after 11 requests at 900 s; b-3 hung on
  its first request for 900 s.
- Ornith failure b-3: its first Read calls used an invented path
  (a home-folder project path that does not exist) instead of the
  working directory, then it tried an Edit built from an imagined file body;
  it found the real file but each request took 100-170 s, so 7 requests hit
  the 900 s cap.
- Timing is contaminated: a video-intelligence ingest shared Ollama during all
  4B runs, and Ollama's 1-minute keep-alive reloaded Ornith from disk at the
  start of most runs. Pass counts and request counts are not affected.
- n = 3 per cell: 5/6 vs 3/6 overall is a direction, not a significant gap.
