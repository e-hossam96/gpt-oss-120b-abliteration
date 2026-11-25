# gpt-oss-120b two-stage abliteration

Removes refusals from `openai/gpt-oss-120b` in two passes with the
[abliterix](https://github.com/wuwangzhang1216/abliterix) **1.12.2** CLI, using a
smaller trial budget than upstream.

This repo has **no abliteration code of its own**. It holds configs, helper scripts and
checks around abliterix.

> [!IMPORTANT]
> - **Results are not directly comparable with upstream.** Upstream's v7 configs do not
>   load in abliterix 1.12.2 (`weight_normalization = "full"` is rejected with the vLLM
>   backend), so both stages use `"pre"`.
> - **Models are exported with upstream's `export_model.py`.** abliterix's built-in
>   export (`--non-interactive-output-dir`) does not work with the vLLM backend.
> - **Check logs, not exit codes.** abliterix 1.12.2 exits 0 on config errors and when a
>   stage has already finished.

## Contents

- [What's in the repo](#whats-in-the-repo)
- [Requirements](#requirements)
- [Local checks (CPU only)](#local-checks-cpu-only)
- [Full run on the GPU machine](#full-run-on-the-gpu-machine)
- [Resuming](#resuming)
- [Evaluation](#evaluation)
- [Heretic cross-check (optional)](#heretic-cross-check-optional)
- [Unverified assumptions](#unverified-assumptions)
- [Known risks](#known-risks)
- [License](#license)

## What's in the repo

| Path | What it does |
| --- | --- |
| `configs/stage1.toml`, `configs/stage2.toml` | Upstream's v7 configs with a few changes ([docs/DEVIATIONS.md](docs/DEVIATIONS.md)) |
| `configs/upstream/` | Upstream's original configs, unmodified |
| `scripts/setup_data.py` | Downloads the prompt datasets into `data/` |
| `scripts/configure_judge.py` | Asks for the LLM judge (name, base URL, API key) |
| `scripts/check_configs.py` | Checks the configs, including with abliterix's own validation |
| `scripts/best_trial.py` | Picks the trial to export |
| `scripts/evaluate.py` | Measures refusals on prompts the search never saw |
| `scripts/heretic_crosscheck.py` | Optional second opinion from Heretic |
| `scripts/upstream/` | Upstream's BF16 conversion and export scripts, unmodified |

Further reading:

- [docs/RECON.md](docs/RECON.md): research notes on abliterix, the datasets and Heretic
- [docs/DEVIATIONS.md](docs/DEVIATIONS.md): every change from upstream, and why
- [docs/UPSTREAM.md](docs/UPSTREAM.md): pinned sources and checksums

## Requirements

| Resource | Needed | Notes |
| --- | --- | --- |
| GPUs | 4× 96 GB (e.g. RTX PRO 6000) | Same as upstream. Configs use `tensor_parallel_size = 4` and 92 GiB `max_memory` per GPU. |
| Disk | 1 TB or more | ~232 GB per BF16 checkpoint (base, Stage 1, Stage 2) plus ~65 GB for the original download |
| RAM | 256 GB or more | vLLM keeps ~20–25 GB of CPU backups per GPU |

`check_configs.py` checks that the GPU count divides the model's 64 attention heads
and 8 KV heads (so 3 GPUs is rejected), and that `max_memory` has one entry per GPU.

**Time** (upstream's estimates, not measured here):

| Step | Time |
| --- | --- |
| BF16 conversion | 8–15 min |
| Stage 1 search (100 trials × ~60 s) | ~1.7 h |
| Stage 2 search (50 trials × ~60 s) | ~50 min |
| Each export (reloads the model, writes ~232 GB) | not given by upstream |

## Local checks (CPU only)

```bash
uv sync
uv run ruff check
uv run pytest -q
uv run python scripts/check_configs.py
uv run python scripts/heretic_crosscheck.py --dry-run
```

Locally, `check_configs.py` only **warns** about things that live on the GPU machine
(the "pod"): downloaded data, model folders and judge variables. On the pod, pass
`--pod` to turn those warnings into errors.

Stage 2's low warm-up count (2.2 trials per search dimension) is always a warning.

## Full run on the GPU machine

Run everything on Linux, **from the repo root**. The final model ends up in `out/stage2/`.

### 0. Install and set up the judge

```bash
uv sync --frozen --extra pod
export HF_TOKEN=...   # read-only; needs access to wangzhang/abliterix-datasets
mkdir -p logs

# Writes the gitignored .env.judge
uv run python scripts/configure_judge.py
```

Run these in **every new shell** to load the judge settings:

```bash
set -a; . ./.env.judge; set +a
JUDGE=(--detection.llm-judge-model "${LLM_JUDGE_MODEL:?}" --detection.llm-judge-base-url "${LLM_JUDGE_BASE_URL:?}")
```

Keep secrets in the environment, never in git.

### 1. Download the data

```bash
uv run python scripts/setup_data.py   # prints only row counts and overlap checks
```

### 2. Convert the model to BF16

The vLLM backend needs a BF16 checkpoint on disk.

```bash
uv run python scripts/upstream/prepare_bf16_checkpoint.py \
  --model openai/gpt-oss-120b --out out/gpt-oss-120b-bf16
uv run python scripts/check_configs.py --pod   # Stage 2's model is still missing: expected
```

### 3. Stage 1: search, then export the best trial

```bash
uv run abliterix --config configs/stage1.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage1.log

TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage1 --model out/gpt-oss-120b-bf16)
uv run python scripts/upstream/export_model.py --model out/gpt-oss-120b-bf16 \
  --checkpoint checkpoints/stage1 --trial "$TRIAL" --config configs/stage1.toml \
  --save-local out/stage1 2>&1 | tee logs/export_stage1.log
```

### 4. Stage 2: search on the Stage 1 model, then the final export

```bash
uv run python scripts/check_configs.py --pod   # must now report 0 errors
uv run abliterix --config configs/stage2.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage2.log

TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage2 --model out/stage1)
uv run python scripts/upstream/export_model.py --model out/stage1 \
  --checkpoint checkpoints/stage2 --trial "$TRIAL" --config configs/stage2.toml \
  --save-local out/stage2 2>&1 | tee logs/export_stage2.log
```

### 5. Heretic cross-check (optional)

Run this **before** step 6 so the evaluation report includes it.

```bash
uv tool install heretic-llm==1.4.0
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2 --dry-run
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2
```

### 6. Evaluate

Uses the unmodified BF16 model as a baseline on the same prompts.

```bash
uv run python scripts/evaluate.py --model out/stage2 --baseline out/gpt-oss-120b-bf16 --tensor-parallel-size 4
```

### Things to know

- **Nothing is uploaded.** `export_model.py` uploads only with `--push-to`, which is
  never passed. `abliterix --non-interactive` exits before its upload menu. Use a
  read-only HF token as a safeguard.
- **Write `--config <path>` with a space.** abliterix 1.12.2 silently ignores
  `--config=<path>`.
- **Check the logs**, not just exit codes (see the warning at the top).

## Resuming

| Situation | What to do |
| --- | --- |
| Search was interrupted | Run the same `abliterix` command again. It continues from `checkpoints/stageN/<model>.jsonl`. |
| Start a stage over | Add `--overwrite-checkpoint`, or delete the journal file. Otherwise a finished stage is skipped (exit code 0). |
| Export failed | Delete the partial `out/stageN` before retrying. |
| Stage 1 was re-exported | Run Stage 2 again from scratch. |

When resuming, settings saved in the journal **override** the config and the command
line, including the judge model and URL. The API key is not saved, so load
`.env.judge` again.

## Evaluation

`scripts/evaluate.py` measures refusals on prompts the search never saw.

| Topic | How it works |
| --- | --- |
| Prompts | Rows of `harmful_1000` that are not in `harmful_500` (ignoring case and whitespace). The search used only `harmful_500`. If the data is missing or no rows are left, the script stops with exit code 2. |
| Generation | vLLM, greedy, at least 256 new tokens (default 256). gpt-oss's reasoning step is skipped so all tokens go to the visible answer. Each model runs in its own process. |
| Judge | The same LLM as the search, with abliterix's judge prompt, one response per request, temperature 0. A test checks that the request matches abliterix's exactly. Requests run 8 at a time; lower `--judge-workers` if the endpoint rate-limits. |
| Broken output | Empty, repeating or mostly-symbol responses count as refusals and are not sent to the judge. |
| Baseline | `--baseline` runs the same prompts on the unmodified model. Use the local BF16 model for a fair comparison. |
| Heretic | If `runs/heretic_crosscheck/summary.json` exists, it is added as a separate section. |

**Output** (prompt text is never printed):

- `runs/eval/results.json` and `runs/eval/summary.md`: refusal count with a 95%
  confidence interval
- `runs/eval/*_records.jsonl`: per-prompt responses

## Heretic cross-check (optional)

`scripts/heretic_crosscheck.py` runs
`heretic --model <base> --evaluate-model <modified>` (Heretic 1.4.0) and saves
`runs/heretic_crosscheck/heretic.log` and `summary.json`.

- **Install separately:** `uv tool install heretic-llm==1.4.0`, or point to it with
  `--heretic-bin`. If Heretic is missing, the script prints `SKIPPED` and exits 0.
- **`--dry-run`** prints the command without needing Heretic or a GPU.

> [!NOTE]
> **Heretic measures something different.** It uses its own prompts
> (`mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, 100 each), detects
> refusals by keyword, generates only 100 tokens, and reports KL divergence on harmless
> prompts. Report its numbers **next to** `evaluate.py`'s, never combined. Agreement is
> a sanity check, not proof.

- **Run it last**, with nothing else on the GPUs. It loads the base model, then the
  modified one (~234 GB), one at a time.
- Needs **PyTorch 2.6 or newer**. Heretic's own install brings a recent version.
- Only **Heretic 1.4.0** output is parsed. Other versions stop with an error and keep
  the log.
- **Choosing `--base`:** `openai/gpt-oss-120b` compares against the original download.
  `out/gpt-oss-120b-bf16` compares against the exact weights abliterix started from.

## Unverified assumptions

| # | Assumption | Safeguard |
| --- | --- | --- |
| 1 | **Dataset access and layout.** `wangzhang/abliterix-datasets` returned HTTP 401 without a token. One JSON file is expected per folder (`good_500`, `harmful_500`, `good_1000`, `harmful_1000`). | `setup_data.py` checks the known checksums of the two 500-row files and fails on anything unexpected. |
| 2 | **Unseen prompts exist.** It is unknown whether `harmful_1000` has rows not in `harmful_500`. Paraphrases are not detected. | `setup_data.py` prints the overlap; `evaluate.py` stops if nothing is left. |
| 3 | **Export works.** abliterix's built-in export is expected to fail with vLLM (from reading the code). Upstream's `export_model.py` should work with 1.12.2 but has not been run. | — |
| 4 | **Export accuracy.** The export bakes the router change into the weights; the search applied it on the fly. The exported model may score differently from the trial. | `evaluate.py` measures the exported model. |
| 5 | **Reasoning during search.** abliterix's vLLM path may not skip gpt-oss's reasoning step, so search-time refusal counts may include it. | `evaluate.py` always skips it. |
| 6 | **vLLM prompts.** gpt-oss's special tokens are passed as text and assumed to be recognized by vLLM. | — |
| 7 | **`weight_normalization = "pre"`** is a fair substitute for `"full"`. Its effect is unmeasured. | — |
| 8 | **Heretic on the GPU machine:** local model paths work, there are no interactive questions, and the original model may or may not stay compressed in memory. | Input is closed, so it fails instead of hanging. |
| 9 | **Judge endpoint** is OpenAI-compatible (`/chat/completions`) and accepts temperature 0 and a JSON-schema `response_format`. Not tested live. `evaluate.py` reuses abliterix's internal judge prompt, which is fixed only by the pinned version. | Use a fixed judge model version so it doesn't change between stages and evaluation. |

## Known risks

- **Unreliable exit codes** in abliterix 1.12.2 (see the warning at the top).
- **Half of upstream's budget in Stage 1:** 100 trials with 40 warm-up trials.
- **Thin warm-up in Stage 2:** 20 warm-up trials for about 9 search dimensions.
- **Stage 2's KL is measured against the Stage 1 model**, not the original. The total
  is only roughly the sum of both stages.
- **Heretic and `evaluate.py` measure different things.** Do not average them.

## License

- This repo: **AGPL-3.0** ([LICENSE](LICENSE)), because it includes abliterix files.
  Attribution is in [NOTICE](NOTICE).
- gpt-oss-120b: **Apache-2.0**. Models derived from it must keep those terms.
- `wangzhang/abliterix-datasets`: license **unknown**.
- Heretic (AGPL-3.0) is only run as an external program.
