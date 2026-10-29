# Deviations from upstream

Baseline: `configs/upstream/gpt_oss_120b_v7s{1,2}.toml` at abliterix
`1940c8c14688772011d5f20b03c5dffb042a3e8f`. Each changed line is also marked
`DEVIATION` inside `configs/stage{1,2}.toml`. Check with
`git diff --no-index configs/upstream/gpt_oss_120b_v7s1.toml configs/stage1.toml`.

## Config values

| Key | Upstream v7s1 | `stage1.toml` | Upstream v7s2 | `stage2.toml` | Reason |
|---|---|---|---|---|---|
| `model.model_id` | `/workspace/gpt-oss-120b-bf16` | `out/gpt-oss-120b-bf16` | `/workspace/gpt-oss-120b-v7s1` | `out/stage1` | Repo-relative paths. Stage 2 must read the Stage 1 export. |
| `optimization.num_trials` | 200 | **100** | 50 | 50 | Limited trial budget. |
| `optimization.num_warmup_trials` | 80 | **40** | 20 | 20 | Limited trial budget. |
| `optimization.sampler_seed` | *(absent)* | **42** (added) | *(absent)* | **42** (added) | Reproducible TPE sampler. The only key not in the upstream files. |
| `optimization.checkpoint_dir` | `checkpoints_gpt_oss_120b_v7s1` | `checkpoints/stage1` | `checkpoints_gpt_oss_120b_v7s2` | `checkpoints/stage2` | Single gitignored folder. |
| `steering.weight_normalization` | `"full"` | **`"pre"`** | `"full"` | **`"pre"`** | **Not a run-specific edit; required.** abliterix 1.12.2 (and master) rejects `"full"` with `backend = "vllm"` at config load: *"backend='vllm' currently uses ProjectionCache, which supports only a 2-D single-direction rank-1 steering tensor. Unsupported rank-k recipe settings: weight_normalization='full'"*. Reproduced on this PC by loading the unmodified upstream file. `"pre"` is the value abliterix's validator substitutes itself when the key is unset under vLLM. This changes the edit (no row-magnitude re-normalisation), so upstream's published v7 numbers are not directly comparable. |
| `[*_prompts].dataset` (4 sections) | `datasets/good_500`, `datasets/harmful_500` | `data/good_500`, `data/harmful_500` | same | same | Data is materialized into gitignored `data/` by `scripts/setup_data.py`. Split and column are unchanged. |
| `model.tensor_parallel_size`, `model.max_memory` | 4, `92GiB` × 4 | unchanged | 4, `92GiB` × 4 | unchanged | Target hardware is 4× 96 GB, as upstream. |

Warm-up budget: Stage 1 has 40 / 9 dims = 4.4 per dim (v5's regime, upstream v7s1 used
8.9). Stage 2 has 20 / 9 = 2.2, unchanged from upstream; `check_configs.py` warns about it.

Comments: both files gained a header naming the source file and SHA, plus `DEVIATION`
notes. In `stage1.toml`, upstream's four-line `[optimization]` comment justifying
200 / 80 trials was replaced by a note on the 100 / 40 budget. The history comments
at the top of both files were kept verbatim, even where they describe upstream's
budgets. No other key, value or section was changed.

## Pipeline (not config)

| Upstream (`deploy_gpt_oss_120b_v7.sh`) | This repo | Reason |
|---|---|---|
| `AX_CONFIG=<cfg> abliterix` | `abliterix --config <cfg> --non-interactive` | Same thing. abliterix 1.12.2 only honours `--config <path>` with a space; `--config=<path>` is silently ignored and falls back to `abliterix.toml`. |
| Export with `scripts/export_model.py` | Same script, vendored unchanged | The brief's `--non-interactive-output-dir` is **not used**. In 1.12.2 that path calls `engine.export_merged()`, which asserts the HF model is a `PeftModel`. With `backend = "vllm"` the CLI unloads the HF model (`engine.model = None`) before the search, and the vLLM in-place edits live in the vLLM workers. The export is therefore expected to fail after the last trial (static reading; not executed). |
| Best trial chosen by inline Python | `scripts/best_trial.py` | Same rule abliterix uses for its own export: min (refusals, kl_divergence) over COMPLETE trials. |
| Judge `google/gemini-3.1-flash-lite-preview` on OpenRouter (TOML value) | Judge supplied by the user at run time: `--detection.llm-judge-model "$LLM_JUDGE_MODEL" --detection.llm-judge-base-url "$LLM_JUDGE_BASE_URL"`, key in `LLM_JUDGE_API_KEY` | The TOML value is kept but overridden. The flags only replace those two fields; every other `[detection]` key still comes from the TOML (checked by parsing the CLI with abliterix 1.12.2). `LLM_JUDGE_API_KEY` is abliterix's own default key variable when a base URL is set. Refusal counts are only comparable with upstream if the same judge is used. |
| `/workspace/...` absolute paths | `out/...` relative to repo root | Run every command from the repo root. |
