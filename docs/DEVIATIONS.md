# Changes from upstream

This page lists every way this repo differs from upstream's v7 pipeline
(abliterix commit `1940c8c14688772011d5f20b03c5dffb042a3e8f`).

- Originals: `configs/upstream/gpt_oss_120b_v7s{1,2}.toml`
- Modified: `configs/stage{1,2}.toml`. Every changed line is marked `DEVIATION`.

To see the full diff:

```bash
git diff --no-index configs/upstream/gpt_oss_120b_v7s1.toml configs/stage1.toml
git diff --no-index configs/upstream/gpt_oss_120b_v7s2.toml configs/stage2.toml
```

## Config values

| Key | Stage 1: upstream → here | Stage 2: upstream → here | Why |
| --- | --- | --- | --- |
| `steering.weight_normalization` | `"full"` → **`"pre"`** | `"full"` → **`"pre"`** | **Required.** See [below](#why-weight_normalization-changed). |
| `optimization.num_trials` | 200 → **100** | 50 (same) | Smaller trial budget |
| `optimization.num_warmup_trials` | 80 → **40** | 20 (same) | Smaller trial budget |
| `optimization.sampler_seed` | *(none)* → **42** | *(none)* → **42** | Reproducible runs. The only added key. |
| `model.model_id` | `/workspace/gpt-oss-120b-bf16` → `out/gpt-oss-120b-bf16` | `/workspace/gpt-oss-120b-v7s1` → `out/stage1` | Paths relative to the repo. Stage 2 reads the Stage 1 export. |
| `optimization.checkpoint_dir` | `checkpoints_gpt_oss_120b_v7s1` → `checkpoints/stage1` | `checkpoints_gpt_oss_120b_v7s2` → `checkpoints/stage2` | One gitignored folder |
| `[*_prompts].dataset` (4 sections) | `datasets/…` → `data/…` | `datasets/…` → `data/…` | `setup_data.py` downloads into the gitignored `data/`. Splits and columns are unchanged. |
| `model.tensor_parallel_size`, `model.max_memory` | 4, 92 GiB × 4 (same) | 4, 92 GiB × 4 (same) | Same hardware as upstream (4× 96 GB) |

### Why `weight_normalization` changed

- abliterix 1.12.2 (and master) **rejects `"full"` with `backend = "vllm"`** when it
  loads the config. This was reproduced with the unmodified upstream file.
- `"pre"` is what abliterix picks itself when the key is unset with vLLM.
- It changes the edit, so **upstream's published results are not directly comparable**.

### Warm-up per search dimension

| Stage | Here | Upstream |
| --- | --- | --- |
| Stage 1 | 40 / 9 = **4.4** | 80 / 9 = 8.9 |
| Stage 2 | 20 / 9 = **2.2** | 20 / 9 = 2.2 |

`check_configs.py` warns about Stage 2's low value.

### Comments in the configs

- Each file starts with a link to its upstream source.
- Each change has a short `DEVIATION` note.
- Upstream's comments were kept, even where they describe upstream's trial budget.
- Nothing else was changed.

## Pipeline

Compared with upstream's `quick_start/deploy_gpt_oss_120b_v7.sh`.

| Step | Upstream | Here | Why |
| --- | --- | --- | --- |
| Pick the config | `AX_CONFIG=<cfg> abliterix` | `abliterix --config <cfg> --non-interactive` | Equivalent. 1.12.2 only accepts `--config <path>` with a space; `--config=<path>` is ignored and it falls back to `abliterix.toml`. |
| Export | `scripts/export_model.py` | Same script, unmodified | abliterix's own export is not used. See [below](#why-abliterixs-own-export-is-not-used). |
| Pick the best trial | Inline Python | `scripts/best_trial.py` | Same rule abliterix uses: fewest refusals, then lowest KL, among completed trials. |
| Judge | `google/gemini-3.1-flash-lite-preview` on OpenRouter, set in the config | Chosen at run time (see [below](#how-the-judge-is-chosen)) | Lets you use any OpenAI-compatible judge |
| Paths | `/workspace/...` (absolute) | `out/...` (relative) | Run every command from the repo root. |

### Why abliterix's own export is not used

`--non-interactive-output-dir` needs the HF model in memory. With the vLLM backend, the
CLI unloads it before the search and never reloads it, so it is expected to fail after
the last trial. (From reading the code, not run.)

### How the judge is chosen

- `--detection.llm-judge-model` and `--detection.llm-judge-base-url` override only those
  two settings (checked against abliterix 1.12.2's CLI parsing). The config value is
  kept but overridden.
- The key goes in `LLM_JUDGE_API_KEY`, the variable abliterix reads when a base URL is
  set.
- **Refusal counts only match upstream's if the same judge is used.**
