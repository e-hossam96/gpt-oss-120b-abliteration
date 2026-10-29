# gpt-oss-120b two-stage abliteration (abliterix)

A limited-trial, two-stage abliteration of `openai/gpt-oss-120b` using
[abliterix](https://github.com/wuwangzhang1216/abliterix) **1.12.2** through its own CLI.
This repo contains no abliteration code. It holds:

- `configs/stage1.toml`, `configs/stage2.toml`: upstream `gpt_oss_120b_v7s1/v7s2.toml` with
  run-specific edits ([docs/DEVIATIONS.md](docs/DEVIATIONS.md)). Unmodified originals are in
  `configs/upstream/`.
- `scripts/setup_data.py`: downloads the prompt datasets into gitignored `data/`.
- `scripts/configure_judge.py`: asks for the LLM judge (name, base URL, API key).
- `scripts/check_configs.py`: config sanity checks, including validation by abliterix itself.
- `scripts/best_trial.py`: picks the trial to export, using abliterix's own selection rule.
- `scripts/upstream/`: upstream's BF16-prep and export scripts, byte-identical.
- `scripts/evaluate.py`: held-out refusal evaluation with an LLM judge.
- `scripts/heretic_crosscheck.py`: optional independent check with Heretic.

Recon findings: [docs/RECON.md](docs/RECON.md). Pinned sources: [docs/UPSTREAM.md](docs/UPSTREAM.md).

> **Read before running.** The unmodified upstream v7 configs do not load in abliterix
> 1.12.2 (or master). `weight_normalization = "full"` is rejected for the vLLM backend,
> so both stage configs use `"pre"`. This changes the edit, and upstream's published v7
> results are not directly comparable. abliterix's `--non-interactive-output-dir`
> export is not used either, because it cannot work with the vLLM in-place backend.
> Winners are exported with upstream's own `export_model.py`. Details are in
> DEVIATIONS.md.

## Quickstart (authoring PC, CPU only)

```
uv sync
uv run ruff check
uv run pytest -q
uv run python scripts/check_configs.py
uv run python scripts/heretic_crosscheck.py --dry-run
```

Locally, `check_configs.py` reports pod-only conditions as warnings: data not
downloaded, model directories missing, LLM judge variables unset. Stage 2's
2.2 warm-up trials per search dimension is also a warning. Use `--pod` on the GPU pod
to make the pod-only checks errors.

## Hardware

GPUs: 4× 96 GB (e.g. RTX PRO 6000), as in upstream's v7 configs:
`tensor_parallel_size = 4` and `max_memory = {0 = "92GiB", 1 = "92GiB", 2 = "92GiB", 3 = "92GiB"}`.
`check_configs.py` checks that TP divides 64 attention heads and 8 KV heads (3 is
rejected) and that `max_memory` has one entry per GPU.

Disk: each BF16 checkpoint is about 232 GB (upstream estimate): base BF16, Stage 1
export and Stage 2 export, plus about 65 GB of MXFP4 originals in the HF cache. Plan
for at least 1 TB. Host RAM: the vLLM in-place editor keeps pinned CPU backups of
roughly 20–25 GB per TP worker (upstream docstring estimate); 256 GB or more is advisable.

## Time estimate (upstream's numbers, not measured here)

Upstream estimates ~60 s per trial on 4× RTX PRO 6000 at TP=4, so Stage 1 (100 trials)
takes about 1.7 h and Stage 2 (50 trials) about 50 min. BF16 prep takes ~8–15 min.
Each `export_model.py` run reloads the model in HF, re-extracts hidden states and writes
~232 GB, with no upstream estimate.

## Pod commands (Linux, run from the repo root)

```bash
# 0. Environment. The pod extra adds vLLM. Secrets go in the environment, never in git.
uv sync --frozen --extra pod
export HF_TOKEN=...            # read-only; needs access to wangzhang/abliterix-datasets (see Unverified)
mkdir -p logs

# LLM judge for both stages and for evaluate.py. Asks for the LLM name, base URL and
# API key (hidden), and writes them to the gitignored .env.judge. Load it in each new shell.
uv run python scripts/configure_judge.py
set -a; . ./.env.judge; set +a
JUDGE=(--detection.llm-judge-model "${LLM_JUDGE_MODEL:?}" --detection.llm-judge-base-url "${LLM_JUDGE_BASE_URL:?}")

# 1. Data (prints row counts and overlap flags only)
uv run python scripts/setup_data.py

# 2. MXFP4 -> BF16 checkpoint with quantization_config stripped (needed by the vLLM in-place path)
uv run python scripts/upstream/prepare_bf16_checkpoint.py --model openai/gpt-oss-120b --out out/gpt-oss-120b-bf16
uv run python scripts/check_configs.py --pod   # stage2's model dir is still missing: expected

# 3. Stage 1 search, then export of the best trial
uv run abliterix --config configs/stage1.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage1.log
TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage1 --model out/gpt-oss-120b-bf16)
uv run python scripts/upstream/export_model.py --model out/gpt-oss-120b-bf16 \
  --checkpoint checkpoints/stage1 --trial "$TRIAL" --config configs/stage1.toml --save-local out/stage1 \
  2>&1 | tee logs/export_stage1.log

# 4. Stage 2 search on the Stage 1 export, then the final export
uv run python scripts/check_configs.py --pod   # must now pass with 0 errors
uv run abliterix --config configs/stage2.toml --non-interactive "${JUDGE[@]}" 2>&1 | tee logs/stage2.log
TRIAL=$(uv run python scripts/best_trial.py --checkpoint-dir checkpoints/stage2 --model out/stage1)
uv run python scripts/upstream/export_model.py --model out/stage1 \
  --checkpoint checkpoints/stage2 --trial "$TRIAL" --config configs/stage2.toml --save-local out/stage2 \
  2>&1 | tee logs/export_stage2.log

# 5. Optional Heretic cross-check (first, so evaluate.py can include it in its report)
uv tool install heretic-llm==1.4.0
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2 --dry-run
uv run python scripts/heretic_crosscheck.py --base openai/gpt-oss-120b --modified out/stage2

# 6. Held-out evaluation, with the unmodified BF16 base on the same prompts
uv run python scripts/evaluate.py --model out/stage2 --baseline out/gpt-oss-120b-bf16 --tensor-parallel-size 4
```

**Local only, no upload.** The final model is `out/stage2/` (gitignored). Nothing above
uploads to the Hugging Face Hub: `export_model.py` uploads only when given `--push-to`,
which is never passed, and `abliterix --non-interactive` returns before the interactive
menu that holds abliterix's upload option. As a backstop, use a fine-grained HF token with
read access only, so an upload would be refused even if one were attempted.

The brief's form `abliterix --config configs/stageN.toml --non-interactive
--non-interactive-output-dir out/stageN` uses real flags. The output-dir export is
dropped for the reason above. The configs already set `non_interactive = true`.
Always write `--config <path>` with a space: 1.12.2 silently ignores `--config=<path>`.

**Check the logs, not only exit codes.** abliterix 1.12.2 exits 0 on config-validation
errors and when a checkpoint is already finished.

## Resuming

- **Interrupted search:** re-run the same `abliterix --config ...` command. In
  non-interactive mode abliterix continues the journal at
  `checkpoints/stageN/<slug(model_id)>.jsonl`. Settings stored in the journal override
  the TOML and the CLI flags, so editing the config does not affect a resumed study.
  That includes the judge model and base URL: a resumed stage keeps its original judge.
  The API key is not stored, so load `.env.judge` again.
- **Restart a stage from scratch:** add `--overwrite-checkpoint`, or delete that journal.
  A finished study is otherwise refused (exit code 0, see above).
- **Export:** `export_model.py` does not check the target directory. Delete a partial
  `out/stageN` before retrying.
- Stage 2 depends on the exact Stage 1 export. If you re-export Stage 1, restart Stage 2.

## Evaluation (`scripts/evaluate.py`)

- **Prompts:** `harmful_1000` rows that do not occur in `harmful_500`, compared after
  case-folding and whitespace normalization. Directions come from `harmful_500`
  `train[:400]` and trial selection from `train[400:]`, so none of these prompts was
  seen. If the data is missing or no row is left, the script stops with a loud error
  (exit 2). Recon could not check the overlap (see Unverified).
- **Generation:** vLLM, greedy, `--max-new-tokens` ≥ 256 (default 256). gpt-oss's
  analysis channel is pre-filled empty, as abliterix's HF path does, so the budget goes
  to the visible answer. Each model runs in its own spawned process.
- **Judge:** the same LLM the search used (`LLM_JUDGE_*` from `.env.judge`), with
  abliterix's judge prompt and JSON schema, batch size 1, temperature 0. A test checks
  that the request body is identical to the one abliterix 1.12.2 sends.
- **Degenerate filter:** empty output, repeated n-grams, symbol runs or single-character
  floods count as refusals and are not sent to the judge.
- **Output:** refusals out of N with a Wilson 95% interval, in `runs/eval/results.json`
  and `runs/eval/summary.md`. Per-prompt records with responses go to
  `runs/eval/*_records.jsonl`; `runs/` is gitignored. Prompt text is never printed.
- `--baseline` runs the identical prompts on the unmodified model. Use the local BF16
  base for a like-for-like comparison.
- If `runs/heretic_crosscheck/summary.json` exists, it is added to the report as a
  separate section.

## Optional Heretic cross-check (`scripts/heretic_crosscheck.py`)

Runs `heretic --model <base> --evaluate-model <modified>` (Heretic 1.4.0) as a
subprocess and writes `runs/heretic_crosscheck/heretic.log` and `summary.json`. The
summary records version, command, models, KL, refusals, prompt total, detector and
datasets. Heretic is never imported, and it is not in `uv.lock`. Install it separately
(`uv tool install heretic-llm==1.4.0`) or pass `--heretic-bin`. If it is missing, the
script prints `SKIPPED` and exits 0. `--dry-run` prints the exact command without
needing Heretic or a GPU.

**This is a different yardstick from `evaluate.py`.** Heretic uses its default
datasets (`mlabonne/harmful_behaviors` and `mlabonne/harmless_alpaca`, `test[:100]`),
a substring refusal detector, 100-token generations, and first-token KL on harmless
prompts. `evaluate.py` uses held-out abliterix prompts, an LLM judge and ≥256 tokens.
Report the two numbers side by side. Never merge them or call them the same metric.
Agreement between them is a sanity check, not proof.

- **Sequencing and memory:** run it after both stages and exports finish, on the same
  pod, with nothing else on the GPUs. It loads both 120B models one after the other: it
  computes the base logprobs, frees the base, then loads the modified BF16 model
  (~234 GB) with `device_map="auto"`. Peak is about one model plus activations.
- **PyTorch:** Heretic's README requires ≥ 2.6 to load MXFP4 gpt-oss (`torch.accelerator`).
  Its own install resolves a newer torch (2.15.0 at recon).
- The log is parsed only if it comes from Heretic 1.4.0, the version whose output format
  was read from source. Any other version, or a missing line, raises an error and
  leaves the raw log.
- `--base openai/gpt-oss-120b` compares against the MXFP4 original. Pass
  `--base out/gpt-oss-120b-bf16` to compare against the same BF16 weights abliterix started from.

## Licensing

This repo is AGPL-3.0 ([LICENSE](LICENSE)) because it includes AGPL-3.0 abliterix files.
Attribution is in [NOTICE](NOTICE). gpt-oss-120b is Apache-2.0, and derived checkpoints
must keep its terms. The licence of `wangzhang/abliterix-datasets` is unknown. Heretic
(AGPL-3.0) is only invoked as an external tool.

## Unverified assumptions (TODO(verify))

1. **Dataset access and layout.** `wangzhang/abliterix-datasets` returned HTTP 401
   without a token. `setup_data.py` assumes `<subset>/*.json` (one file per subset:
   `good_500`, `harmful_500`, `good_1000`, `harmful_1000`). It verifies the two upstream
   SHA256 pins for the 500 sets and fails loudly on any other layout.
2. **Held-out feasibility.** Whether `harmful_1000` contains rows outside `harmful_500`,
   and whether either 1000 set is a superset of its 500 set, is unknown.
   `setup_data.py` prints the flags, and `evaluate.py` stops if nothing is held out.
   Near-duplicates such as paraphrases or translations are not detected.
3. **Export path.** `--non-interactive-output-dir` is expected to fail with the vLLM
   in-place backend (static reading of `cli.py` and `engine.export_merged`).
   `scripts/upstream/export_model.py` is assumed to work with 1.12.2 because all APIs
   it calls exist, but it has not been run.
4. **Export fidelity.** The HF export applies router suppression by editing gate weight
   rows. The vLLM search applied router bias live. Whether `out/stageN` reproduces the
   scored trial's refusals and KL is unverified, so `evaluate.py` measures the exported
   weights.
5. **Analysis channel during search.** abliterix's vLLM generator does not visibly apply
   the gpt-oss CoT-skip prefix that its HF path uses. Trial-time refusal counts may be
   computed on text that includes the analysis channel. `evaluate.py` always skips it.
6. **vLLM prompt handling in `evaluate.py`.** The prompt is the chat-template string plus
   Harmony special tokens as text. vLLM is assumed to tokenize these as special tokens.
7. **`weight_normalization = "pre"`** is assumed to be an acceptable stand-in for upstream's
   `"full"`. Its effect on refusals and KL for this model is unmeasured.
8. **Heretic on the pod:** local-directory arguments, no interactive prompt in evaluate
   mode (stdin is closed so it fails rather than hangs), and whether the MXFP4 base
   stays quantized or is dequantized in memory.
9. **Judge endpoint:** the user-supplied LLM must serve an OpenAI-compatible
   `/chat/completions` that accepts temperature 0 and a JSON-schema `response_format`
   (abliterix's defaults for these). Not tested against a live endpoint. Use a model
   pinned by version, so it does not change between Stage 1, Stage 2 and evaluation.
   `evaluate.py` reuses abliterix's private `_JUDGE_PROMPT` and `_JUDGE_RESPONSE_FORMAT`,
   which are pinned only by the abliterix version.

## Known risks

- abliterix 1.12.2 exit codes are unreliable (see above). Master fixes this but was not
  pinned, because 1.12.2 already supports every key the configs use.
- 100 / 40 trials in Stage 1 is half of upstream's v7s1 budget. Stage 2 keeps 20 warm-up
  trials for ~9 dimensions, which is thin for TPE.
- Stage 2 KL is measured against Stage 1, not the original. Total drift is only
  approximately additive (upstream comment).
- The two Heretic and `evaluate.py` numbers measure different things. Do not average them.
