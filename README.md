# gpt-oss-120b two-stage abliteration

Removes refusals from `openai/gpt-oss-120b` in two passes, using the
[abliterix](https://github.com/wuwangzhang1216/abliterix) **1.12.2** command-line tool
with a reduced trial budget. This repo contains no abliteration code of its own, only
configs and helper scripts:

| Path | Purpose |
| --- | --- |
| `configs/stage1.toml`, `configs/stage2.toml` | Upstream's v7 configs with a few changes ([docs/DEVIATIONS.md](docs/DEVIATIONS.md)). Originals are in `configs/upstream/`. |
| `scripts/setup_data.py` | Downloads the prompt datasets into `data/`. |
| `scripts/configure_judge.py` | Asks for the LLM judge (name, base URL, API key). |
| `scripts/check_configs.py` | Checks the configs, including with abliterix's own validation. |
| `scripts/best_trial.py` | Picks the trial to export. |
| `scripts/upstream/` | Upstream's BF16 conversion and export scripts, unmodified. |
| `scripts/evaluate.py` | Measures refusals on prompts the search never saw. |
| `scripts/heretic_crosscheck.py` | Optional second opinion from Heretic. |

More detail: [docs/RECON.md](docs/RECON.md) (findings) and
[docs/UPSTREAM.md](docs/UPSTREAM.md) (pinned sources).

> **Read before running.** Upstream's v7 configs do not load in abliterix 1.12.2:
> `weight_normalization = "full"` is rejected with the vLLM backend, so both configs use
> `"pre"`. Results are therefore not directly comparable with upstream's. abliterix's
> built-in export (`--non-interactive-output-dir`) does not work with the vLLM backend,
> so models are exported with upstream's `export_model.py` instead.

## Local checks (CPU only)

```bash
uv sync
uv run ruff check
uv run pytest -q
uv run python scripts/check_configs.py
uv run python scripts/heretic_crosscheck.py --dry-run
```

Locally, `check_configs.py` only warns about things that exist on the GPU machine
("pod"): downloaded data, model directories and judge variables. Pass `--pod` there to
turn those warnings into errors. Stage 2's low warm-up count (2.2 trials per search
dimension) is always a warning.

## Hardware

- **GPUs:** 4× 96 GB (e.g. RTX PRO 6000), as upstream. The configs use
  `tensor_parallel_size = 4` and `max_memory` of 92 GiB per GPU. `check_configs.py`
  checks that the GPU count divides the model's 64 attention heads and 8 KV heads
  (so 3 is rejected) and that `max_memory` has one entry per GPU.
- **Disk:** at least 1 TB. Each BF16 checkpoint is about 232 GB (base, Stage 1, Stage 2),
  plus about 65 GB for the original download.
- **RAM:** 256 GB or more. vLLM keeps about 20–25 GB of CPU backups per GPU.

## Time (upstream's estimates, not measured)

About 60 s per trial: Stage 1 (100 trials) takes about 1.7 h and Stage 2 (50 trials)
about 50 min. BF16 conversion takes 8–15 min. Each export reloads the model and writes
about 232 GB; upstream gives no time for it.

## Running on the GPU machine (Linux, from the repo root)

```bash
# 0. Install. Keep secrets in the environment, never in git.
uv sync --frozen --extra pod
export HF_TOKEN=...            # read-only; needs access to wangzhang/abliterix-datasets
mkdir -p logs

# Set up the LLM judge (writes the gitignored .env.judge). Load it in every new shell.
uv run python scripts/configure_judge.py
set -a; . ./.env.judge; set +a
JUDGE=(--detection.llm-judge-model "${LLM_JUDGE_MODEL:?}" --detection.llm-judge-base-url "${LLM_JUDGE_BASE_URL:?}")

# 1. Data (prints only row counts and overlap checks)
uv run python scripts/setup_data.py

# 2. Convert the model to BF16 (required by the vLLM backend)
uv run python scripts/upstream/prepare_bf16_checkpoint.py --model openai/gpt-oss-120b --out out/gpt-oss-120b-bf16
uv run python scripts/check_configs.py --pod   # Stage 2's model is still missing: expected

# 3. Stage 1 search, then export the best trial
uv run abliterix --config configs/stage1.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage1.log
TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage1 --model out/gpt-oss-120b-bf16)
uv run python scripts/upstream/export_model.py --model out/gpt-oss-120b-bf16 \
  --checkpoint checkpoints/stage1 --trial "$TRIAL" --config configs/stage1.toml --save-local out/stage1 \
  2>&1 | tee logs/export_stage1.log

# 4. Stage 2 search on the Stage 1 model, then the final export
uv run python scripts/check_configs.py --pod   # must now report 0 errors
uv run abliterix --config configs/stage2.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage2.log
TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage2 --model out/stage1)
uv run python scripts/upstream/export_model.py --model out/stage1 \
  --checkpoint checkpoints/stage2 --trial "$TRIAL" --config configs/stage2.toml --save-local out/stage2 \
  2>&1 | tee logs/export_stage2.log

# 5. Optional Heretic check (run before step 6 so the report includes it)
uv tool install heretic-llm==1.4.0
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2 --dry-run
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2

# 6. Evaluate, with the unmodified BF16 model as a baseline on the same prompts
uv run python scripts/evaluate.py --model out/stage2 --baseline out/gpt-oss-120b-bf16 --tensor-parallel-size 4
```

The final model is in `out/stage2/`. **Nothing is uploaded.** `export_model.py` uploads
only with `--push-to`, which is never passed, and `abliterix --non-interactive` exits
before its upload menu. As a safeguard, use a read-only HF token.

Notes:

- Write `--config <path>` with a space. abliterix 1.12.2 silently ignores `--config=<path>`.
- **Check the logs, not just exit codes.** abliterix 1.12.2 exits 0 on config errors and
  when a stage has already finished.

## Resuming

- **Interrupted search:** run the same `abliterix` command again. It continues from
  `checkpoints/stageN/<model>.jsonl`. Settings saved there take priority over the config
  and command line, including the judge model and URL. The API key is not saved, so
  load `.env.judge` again.
- **Start a stage over:** add `--overwrite-checkpoint` or delete the journal file.
  Otherwise a finished stage is skipped (with exit code 0).
- **Failed export:** delete the partial `out/stageN` before retrying.
- If you re-export Stage 1, run Stage 2 again from scratch.

## Evaluation (`scripts/evaluate.py`)

- **Prompts:** rows of `harmful_1000` that are not in `harmful_500` (ignoring case and
  whitespace). The search only used `harmful_500`, so these are unseen. If the data is
  missing or no rows are left, the script stops with exit code 2.
- **Generation:** vLLM, greedy, at least 256 new tokens (default 256). gpt-oss's
  reasoning step is skipped so all tokens go to the visible answer. Each model runs in
  its own process.
- **Judge:** the same LLM as the search, with abliterix's judge prompt, one response per
  request, temperature 0. A test checks that the request matches abliterix's exactly.
- **Broken output** (empty, repeating, or mostly symbols) counts as a refusal and is not
  sent to the judge.
- **Results:** refusal count with a 95% confidence interval in `runs/eval/results.json`
  and `runs/eval/summary.md`. Per-prompt responses are in `runs/eval/*_records.jsonl`.
  Prompt text is never printed.
- `--baseline` runs the same prompts on the unmodified model. Use the local BF16 model
  for a fair comparison.
- If `runs/heretic_crosscheck/summary.json` exists, it is added as a separate section.

## Heretic check (`scripts/heretic_crosscheck.py`, optional)

Runs `heretic --model <base> --evaluate-model <modified>` (Heretic 1.4.0) and saves
`runs/heretic_crosscheck/heretic.log` and `summary.json`. Heretic is a separate install
(`uv tool install heretic-llm==1.4.0` or `--heretic-bin`); if it is missing the script
prints `SKIPPED` and exits 0. `--dry-run` prints the command without needing Heretic or
a GPU.

**Heretic measures something different.** It uses its own prompts
(`mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`, 100 each), detects refusals
by keyword, generates only 100 tokens, and reports KL divergence on harmless prompts.
Report its numbers next to `evaluate.py`'s, never combined. Agreement is a sanity
check, not proof.

- Run it last, with nothing else on the GPUs. It loads the base model, then the
  modified one (~234 GB), one at a time.
- It needs PyTorch 2.6 or newer; its own install brings a recent version.
- Only Heretic 1.4.0 output is parsed. Other versions stop with an error and keep the log.
- `--base openai/gpt-oss-120b` compares against the original download. Use
  `--base out/gpt-oss-120b-bf16` to compare against the exact weights abliterix started from.

## License

AGPL-3.0 ([LICENSE](LICENSE)), because it includes abliterix files. Attribution is in
[NOTICE](NOTICE). gpt-oss-120b is Apache-2.0, and models derived from it must keep
those terms. The license of `wangzhang/abliterix-datasets` is unknown. Heretic
(AGPL-3.0) is only run as an external program.

## Unverified assumptions

1. **Dataset access.** `wangzhang/abliterix-datasets` returned HTTP 401 without a token.
   `setup_data.py` expects one JSON file per folder (`good_500`, `harmful_500`,
   `good_1000`, `harmful_1000`), checks the known checksums of the two 500-row files,
   and fails on anything else.
2. **Unseen prompts.** It is unknown whether `harmful_1000` has rows not in
   `harmful_500`. `setup_data.py` prints the overlap and `evaluate.py` stops if nothing
   is left. Paraphrases are not detected.
3. **Export.** abliterix's built-in export is expected to fail with vLLM (from reading
   the code). Upstream's `export_model.py` should work with 1.12.2 but has not been run.
4. **Export accuracy.** The export bakes the router change into the weights, while the
   search applied it on the fly. The exported model may score differently from the
   trial, so `evaluate.py` measures the exported model.
5. **Reasoning during search.** abliterix's vLLM path may not skip gpt-oss's reasoning
   step, so search-time refusal counts may include it. `evaluate.py` always skips it.
6. **vLLM prompts.** `evaluate.py` passes gpt-oss's special tokens as text and assumes
   vLLM recognizes them.
7. **`weight_normalization = "pre"`** is assumed to be a fair substitute for `"full"`.
   Its effect is unmeasured.
8. **Heretic on the GPU machine:** local model paths, no interactive questions (input is
   closed so it fails instead of hanging), and whether the original model stays
   compressed in memory.
9. **Judge endpoint:** must be OpenAI-compatible (`/chat/completions`) and accept
   temperature 0 and a JSON-schema `response_format`. Not tested against a live
   endpoint. Use a fixed model version so it doesn't change between stages and
   evaluation. `evaluate.py` reuses abliterix's internal judge prompt, which is only
   fixed by the pinned abliterix version.

## Known risks

- abliterix 1.12.2 exit codes are unreliable (see above).
- Stage 1 runs 100 trials with 40 warm-up trials, half of upstream's budget. Stage 2
  has only 20 warm-up trials for about 9 search dimensions, which is thin.
- Stage 2's KL is measured against the Stage 1 model, not the original. The total is
  only roughly the sum of both stages.
- Heretic and `evaluate.py` measure different things. Do not average them.
