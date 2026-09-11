# Raw mask benchmark failures — 2026-09-11

Captured while running the unchanged `qa-mask-bench` inferlet from
`feat/mask-bench` commit `d250e60b8199c6ba48bd8a62b61caa373433d68e`.

- Models: Qwen3-Coder-30B-A3B-Instruct Q5_K_M and Qwen2.5-7B-Instruct Q5_K_M.
- Server: installed PIE 0.4.0 (`/workspace/pie-runtime/bin/pie`), portable/CUDA
  backend on an NVIDIA H100 80GB HBM3; not a newly built branch server.
- KV page size: 32; total pages: 1024; output cap: 64 tokens.
- Dataset: this inferlet's `rounds.json`, five rounds with six documents and
  five two-question asks per round. Cases ran sequentially through a persistent
  server, rebuilding the base cache for each round/case invocation.

The `*-errors.jsonl` files preserve the runner's raw error records: 20 failed
invocations per model, covering cases 2, 3, 5, and 6 in every round. These cases
returned no completed ask results. Cases 1 and 4 completed with 100% exact match
on 50 questions each, for both models.

The `*-server.log` files are unmodified server-output snapshots containing the
underlying token-position/sequence-length errors. The A3B log has 21 planner
errors because round 0, case 2 was attempted once before the full matrix was
resumed; its JSONL contains the 20 matrix failures. The Qwen2.5 log has 20.

These records capture runtime execution failures, not incorrect model answers.
No valid masking latency or accuracy measurement is available for failed cases.
