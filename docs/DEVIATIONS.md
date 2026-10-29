# Changes from upstream

Compared with `configs/upstream/gpt_oss_120b_v7s{1,2}.toml` (abliterix commit
`1940c8c14688772011d5f20b03c5dffb042a3e8f`). Every changed line is marked `DEVIATION`
in `configs/stage{1,2}.toml`. To see the full diff:
`git diff --no-index configs/upstream/gpt_oss_120b_v7s1.toml configs/stage1.toml`.

## Config values

| Key | Upstream Stage 1 | `stage1.toml` | Upstream Stage 2 | `stage2.toml` | Reason |
| --- | --- | --- | --- | --- | --- |
| `model.model_id` | `/workspace/gpt-oss-120b-bf16` | `out/gpt-oss-120b-bf16` | `/workspace/gpt-oss-120b-v7s1` | `out/stage1` | Paths relative to the repo. Stage 2 reads the Stage 1 export. |
| `optimization.num_trials` | 200 | **100** | 50 | 50 | Smaller trial budget. |
| `optimization.num_warmup_trials` | 80 | **40** | 20 | 20 | Smaller trial budget. |
| `optimization.sampler_seed` | *(none)* | **42** | *(none)* | **42** | Reproducible runs. The only added key. |
| `optimization.checkpoint_dir` | `checkpoints_gpt_oss_120b_v7s1` | `checkpoints/stage1` | `checkpoints_gpt_oss_120b_v7s2` | `checkpoints/stage2` | One gitignored folder. |
| `steering.weight_normalization` | `"full"` | **`"pre"`** | `"full"` | **`"pre"`** | **Required, not a choice.** abliterix 1.12.2 (and master) rejects `"full"` with `backend = "vllm"` when loading the config (reproduced with the unmodified upstream file). `"pre"` is what abliterix uses itself when the key is unset with vLLM. It changes the edit, so upstream's published results are not directly comparable. |
| `[*_prompts].dataset` (4 sections) | `datasets/good_500`, `datasets/harmful_500` | `data/good_500`, `data/harmful_500` | same | same | `scripts/setup_data.py` downloads into the gitignored `data/`. Splits and columns are unchanged. |
| `model.tensor_parallel_size`, `model.max_memory` | 4, 92 GiB × 4 | unchanged | 4, 92 GiB × 4 | unchanged | Same hardware as upstream (4× 96 GB). |

Warm-up per search dimension: Stage 1 has 40 / 9 = 4.4 (upstream had 8.9). Stage 2 has
20 / 9 = 2.2, the same as upstream; `check_configs.py` warns about it.

Comments: each file starts with a link to its upstream source, and each change has a
short `DEVIATION` note. Upstream's comments were kept, even where they describe
upstream's trial budget. Nothing else was changed.

## Pipeline

| Upstream (`deploy_gpt_oss_120b_v7.sh`) | This repo | Reason |
| --- | --- | --- |
| `AX_CONFIG=<cfg> abliterix` | `abliterix --config <cfg> --non-interactive` | Equivalent. 1.12.2 only accepts `--config <path>` with a space; `--config=<path>` is ignored and it falls back to `abliterix.toml`. |
| Export with `scripts/export_model.py` | Same script, unmodified | abliterix's own `--non-interactive-output-dir` export is **not used**. It needs the HF model in memory, but with the vLLM backend the CLI unloads it before the search. It is expected to fail after the last trial (from reading the code, not run). |
| Best trial picked by inline Python | `scripts/best_trial.py` | Same rule abliterix uses: fewest refusals, then lowest KL, among completed trials. |
| Judge `google/gemini-3.1-flash-lite-preview` on OpenRouter (in the config) | Judge chosen at run time with `--detection.llm-judge-model` and `--detection.llm-judge-base-url`; key in `LLM_JUDGE_API_KEY` | The config value is kept but overridden. The flags replace only those two settings (checked against abliterix 1.12.2's CLI parsing). `LLM_JUDGE_API_KEY` is the variable abliterix reads when a base URL is set. Refusal counts only match upstream's if the same judge is used. |
| `/workspace/...` absolute paths | `out/...` relative paths | Run every command from the repo root. |
