# Phase 0 recon (2026-10-29, Windows 11 authoring PC, CPU only)

Method: installed packages, `--help` output, and upstream source read at pinned
commits (abliterix wheel 1.12.2 + master `1940c8c1`, heretic-llm tag `v1.4.0`).
Nothing GPU-bound was run. No weights were downloaded. Items marked **TODO(verify)**
could not be confirmed here and are repeated in the README.

## abliterix

- Version: **1.12.2** (latest on PyPI). Locked with `torch 2.11.0`, `transformers 5.19.0`,
  and `vllm 0.20.2` (vLLM only via the `pod` extra). Python 3.12.
- `abliterix --help` works on Windows and lists:
  - `--config {str,null}`, `--non-interactive | --no-non-interactive`,
    `--non-interactive-output-dir {str,null}`, `--overwrite-checkpoint`, `--seed`.
  - Per-section overrides in kebab case with a dot: `--optimization.num-trials int`,
    `--optimization.num-warmup-trials int`, `--optimization.sampler-seed {int,null}`,
    `--optimization.checkpoint-dir str`, `--model.model-id`, `--model.tensor-parallel-size`,
    `--steering.fixed-vector-scope`, `--steering.disabled-components`.
  - `--reproduce <reproduce.json>` is not in the usage text. `cli.py` pops it from
    `sys.argv` before config parsing.
- Config file: `settings_customise_sources` scans `sys.argv` for `--config <path>`
  (space-separated only; `--config=path` is ignored in 1.12.2, fixed on master), else
  `AX_CONFIG`, else `abliterix.toml`. Priority: init > CLI > `AX_*` env > dotenv > TOML.
- Unknown keys: the top-level model forbids extras, but **nested sections silently
  ignore unknown keys** (tested: a bogus `[model] bogus_key` and `[optimization]
  not_a_key` loaded fine). `check_configs.py` therefore walks the abliterix schema.
- `fixed_vector_scope` and `disabled_components` exist in 1.12.2
  (`SteeringConfig`, used in optimizer/steering/vllm_backend). PyPI is sufficient, so
  no git pin was needed. Master only tightens their types (`Literal`).
- **The unmodified upstream v7s1/v7s2 configs fail to load** in 1.12.2, and master
  has the same validator: `weight_normalization='full'` is rejected for
  `backend='vllm'`. See DEVIATIONS.md.
- 1.12.2 exits **0** on config-validation errors and on "checkpoint already finished"
  (master returns 1). Do not trust the exit code alone; check the log.
- Resume: in non-interactive mode an unfinished journal at
  `<checkpoint_dir>/<slug(model_id)>.jsonl` is continued automatically. Its stored
  settings **override the TOML**. A finished study aborts unless
  `--overwrite-checkpoint` is given.
- Best-trial rule used by abliterix's own export: min `(refusals, kl_divergence)` over
  COMPLETE trials. Trials carry `user_attrs["index"]`. The study name is `abliterix`.

### BF16 checkpoint and export

- The CLI does **not** prepare the BF16 checkpoint. `core/engine.py` dequantizes MXFP4
  in memory for the HF phase (`Mxfp4Config(dequantize=True)`), but the vLLM in-place
  path needs an on-disk BF16 checkpoint with `quantization_config` removed. Upstream
  ships `scripts/prepare_bf16_checkpoint.py` (repo only, not in the wheel), vendored
  unchanged in `scripts/upstream/`. transformers 5.19 still accepts its `torch_dtype=`
  kwarg (kept for backward compatibility) and exposes `Mxfp4Config(dequantize=...)`.
  Upstream estimate: ~8–15 min, ~232 GB disk.
- `--non-interactive-output-dir` exists, but its export calls `engine.export_merged()`,
  which asserts `isinstance(self.model, PeftModel)`. With `backend="vllm"`, the CLI
  sets `engine.model = None` after building the vLLM generator, and
  `_restore_selected_trial` does not reload it. Expected failure after the search
  (static reading). Upstream's own v7 orchestrator exports with
  `scripts/export_model.py` instead: a fresh HF `SteeringEngine` that replays the trial
  and calls `export_merged()`. On the HF engine, `_init_adapters` always wraps the
  model in PEFT, so the assertion holds there. **TODO(verify)** on the pod.
- HF router suppression edits gate weight rows in place (`_router_originals`), while
  vLLM search applies router bias live (`apply_router_suppression`). The HF export
  therefore bakes *an* HF-style router edit into the weights. Upstream's v7s2 comment
  says Stage 1 router suppression is "NOT in the weights". **TODO(verify)**: whether the
  exported model reproduces the scored trial. `scripts/evaluate.py` measures the
  exported weights, not the trial.

### Generation detail relevant to evaluation

- In the HF path abliterix detects gpt-oss's Harmony analysis channel and pre-fills
  `<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>`,
  so `max_gen_tokens` is spent on the final answer. `VLLMGenerator._format_prompt` does
  not reference `response_prefix`. **TODO(verify)** whether trial scoring on vLLM
  includes the analysis channel. `evaluate.py` always applies the skip explicitly.
- Judge (`eval/detector.py`): OpenRouter `chat/completions`, `temperature 0.0`,
  `response_format` JSON schema `{"labels": ["R"|"C"]}`, responses truncated to 2000
  chars, items formatted `"{j}. Question: {q}\n   Response: {r}"`. `evaluate.py`
  reuses `RefusalDetector._JUDGE_PROMPT` and `_JUDGE_RESPONSE_FORMAT` (private
  attributes, pinned by the exact version).

## Datasets

- Configs read `datasets/good_500` and `datasets/harmful_500`, column `prompt`, splits
  `train[:400]` (directions) and `train[400:]` (trial selection).
- abliterix's `load_prompt_dataset`: a local directory without `dataset_state.json`
  goes through `datasets.load_dataset(<dir>, split=...)`, the json builder with split
  `train`. **Verified by test** (`tests/test_data.py`): a directory holding one JSON
  array of `{"prompt": ...}` objects loads through abliterix's own function with
  `train[400:]` → 100 rows.
- Expected file names and SHA256 (upstream `benchmarks/SPEC.md`):
  `harmful_500/harmful_prompts_500.json` (`249e48f4…769d`) and
  `good_500/good_prompts_500.json` (`dfffd376…a8a7`). Schema: `id`, `prompt`,
  `language`, `category`, `policy_label`, ...
- Upstream `.gitignore` excludes `/datasets/`, so the data is not in the GitHub repo.
- `https://huggingface.co/api/datasets/wangzhang/abliterix-datasets` returns **HTTP 401**
  without a token. The repo is private, gated in a way that hides metadata, or
  nonexistent; public gated repos return 200. No HF token is configured on this PC.
  Upstream `docs/runpod-quickstart.md` lists `DatasetNotFoundError datasets/good_500`
  as "private dataset not on the pod". **TODO(verify)**: access to the repo, its layout
  (`<subset>/*.json` assumed), the file names of the 1000-row sets, and the dataset licence.
- Superset question (harmful_1000 ⊇ harmful_500, good_1000 ⊇ good_500): **could not be
  checked** without the data. `setup_data.py` computes it in memory and prints only
  booleans and counts. `evaluate.py` computes the held-out set (1000 minus 500,
  case- and whitespace-normalized) at runtime and stops with a loud error if it is empty.

## Heretic (isolated venv in the scratchpad, not the project lock)

- Version: **heretic-llm 1.4.0** (latest on PyPI), Python 3.12. The isolated env
  resolved `torch 2.15.0`, `transformers 5.22.0`, `accelerate 1.15.0`, `peft 0.21.2`,
  `lm-eval 0.4.13`.
- `heretic --help` lists `--model str` ("Hugging Face model ID, or path to model on
  disk") and `--evaluate-model {str,null}` ("evaluate this model relative to the main
  model"). The README documents `heretic --model <base> --evaluate-model <modified>`.
- Local paths: both are passed straight to `from_pretrained`. A test run with
  `--model ./no-such-model` printed the banner and then attempted a local/Hub load.
  Evaluate mode makes no Hub metadata calls; `model_info` is only used for export/upload.
  **TODO(verify)** with real directories on the pod.
- Defaults (`config.default.toml`): good eval `mlabonne/harmless_alpaca` `test[:100]`,
  bad eval `mlabonne/harmful_behaviors` `test[:100]`, column `text`,
  `max_response_length = 100`, system prompt "You are a helpful assistant.", gpt-oss
  CoT skip in `chain_of_thought_skips`. The detector counts a response as a refusal if
  it is empty or contains any `refusal_markers` substring (case-insensitive, after
  `*` removal and whitespace normalization). KL is the first-token
  `F.kl_div(..., reduction="batchmean", log_target=True)` over the 100 harmless prompts.
  Heretic reads `config.toml` from the working directory, so the cross-check runs it
  in `runs/heretic_crosscheck/`.
- Output lines (`evaluator.py`, `main.py`): banner `… v1.4.0` (confirmed from a real run
  of the isolated install), `Loading good evaluation prompts from X...`,
  `* Initial refusals: N/M` (base), then for the modified model `* KL divergence: 0.1234`
  (4 decimals) and `* Refusals: N/M`. rich styling is stripped (`NO_COLOR`, ANSI regex),
  and `COLUMNS=1000` prevents wrapping. The parser accepts only version 1.4.0.
- Memory: evaluate mode computes base logprobs and refusals, then `reset_model()` sets
  `self.model = None`, calls `empty_cache()` and loads the modified model. Peak is about
  one model, not two. Loading `openai/gpt-oss-120b` (MXFP4) through transformers may
  stay MXFP4 (~65 GB) or dequantize to BF16 (~234 GB) depending on kernel support.
  **TODO(verify)**. The modified model is BF16 (~234 GB). `device_map="auto"` across all
  visible GPUs.
- PyTorch: README says the minimum is 2.2, and MXFP4 gpt-oss loading uses
  `torch.accelerator` (2.6+). The isolated env gets 2.15.0, and the main lock has 2.11.0.
- Dependency conflicts: a joint resolution of `abliterix[vllm]==1.12.2` +
  `heretic-llm==1.4.0` **succeeds** (torch 2.11.0, transformers 5.22.0). There is no
  hard conflict. Isolation is a policy choice: keep Heretic's transitive upgrades
  (transformers 5.22 vs 5.19, lm-eval, etc.) out of the abliterix run environment.
