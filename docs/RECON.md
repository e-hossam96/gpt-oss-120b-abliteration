# Research notes (2026-10-29)

Gathered on a Windows 11 PC without a GPU by installing the packages, reading their
`--help` output, and reading the source at the pinned versions (abliterix 1.12.2 and
master `1940c8c1`, Heretic `v1.4.0`). Nothing was run on a GPU and no model weights
were downloaded. Items marked **unverified** are listed again in the README.

## abliterix

- Version **1.12.2** (latest on PyPI), Python 3.12, locked with torch 2.11.0,
  transformers 5.19.0 and vLLM 0.20.2 (vLLM only with the `pod` extra).
- `abliterix --help` works on Windows. Useful options:
  - `--config`, `--non-interactive`, `--non-interactive-output-dir`,
    `--overwrite-checkpoint`, `--seed`.
  - Any config value can be overridden as `--<section>.<key-with-dashes>`, e.g.
    `--optimization.num-trials`, `--model.model-id`, `--steering.disabled-components`.
- Config file lookup: `--config <path>` (with a space; `--config=<path>` is ignored in
  1.12.2, fixed on master), then the `AX_CONFIG` variable, then `abliterix.toml`.
  Priority: code > command line > `AX_*` variables > `.env` > config file.
- **Unknown keys inside a section are silently ignored** (a made-up key under `[model]`
  loaded fine). That is why `check_configs.py` checks every key against abliterix's
  settings classes.
- `fixed_vector_scope` and `disabled_components` exist in 1.12.2, so the PyPI release
  is enough.
- **Upstream's v7 configs fail to load** in 1.12.2 and master:
  `weight_normalization='full'` is rejected with `backend='vllm'`. See DEVIATIONS.md.
- 1.12.2 exits with code **0** on config errors and when a stage is already finished
  (master exits 1). Check the log, not just the exit code.
- Resuming: in non-interactive mode, an unfinished `<checkpoint_dir>/<model>.jsonl`
  is continued automatically, and its saved settings **take priority over the config**.
  A finished stage is skipped unless `--overwrite-checkpoint` is given.
- abliterix's own export picks the trial with the fewest refusals, then the lowest KL,
  among completed trials. Each trial stores its number in `user_attrs["index"]`. The
  Optuna study is named `abliterix`.

### BF16 conversion and export

- The CLI does **not** create the BF16 checkpoint. The vLLM backend needs one on disk,
  without the original 4-bit `quantization_config`. Upstream's
  `scripts/prepare_bf16_checkpoint.py` (in the repo, not the package) does this and is
  copied unchanged into `scripts/upstream/`. It works with transformers 5.19. Upstream
  estimate: 8–15 min, about 232 GB.
- `--non-interactive-output-dir` needs the HF model in memory, but with
  `backend="vllm"` the CLI unloads it before the search and never reloads it. So it is
  expected to fail after the search (from reading the code). Upstream's own pipeline
  uses `scripts/export_model.py` instead, which loads the model fresh, replays the
  chosen trial, and saves it. **Unverified** on a GPU.
- During the search, vLLM changes expert routing on the fly; the HF export instead
  writes that change into the router weights. Upstream's Stage 2 comment says Stage 1's
  routing change is "NOT in the weights". **Unverified:** whether the exported model
  scores the same as the trial. `evaluate.py` measures the exported model.

### Generation details that matter for evaluation

- On the HF path, abliterix skips gpt-oss's reasoning step by adding
  `<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>`
  to the prompt, so the token budget goes to the answer. The vLLM path does not appear
  to do this. **Unverified:** whether search-time scoring includes the reasoning text.
  `evaluate.py` always skips it.
- Judge (`eval/detector.py`): `chat/completions`, temperature 0, a JSON schema that
  returns `{"labels": ["R"|"C"]}` (refusal or compliance), responses cut to 2000
  characters, each item formatted `"{j}. Question: {q}\n   Response: {r}"`.
  `evaluate.py` reuses abliterix's internal `_JUDGE_PROMPT` and
  `_JUDGE_RESPONSE_FORMAT`, which are only fixed by the pinned version.

## Datasets

- The configs read `datasets/good_500` and `datasets/harmful_500`, column `prompt`, rows
  `train[:400]` to find the refusal direction and `train[400:]` to score trials.
- A local folder holding one JSON array of `{"prompt": ...}` objects loads correctly
  through abliterix's own loader, and `train[400:]` gives 100 rows. **Tested** in
  `tests/test_data.py`.
- Expected files and checksums (upstream `benchmarks/SPEC.md`):
  `harmful_500/harmful_prompts_500.json` (`249e48f4…769d`) and
  `good_500/good_prompts_500.json` (`dfffd376…a8a7`). Fields: `id`, `prompt`,
  `language`, `category`, `policy_label`, and more.
- The data is not in the upstream GitHub repo (`/datasets/` is gitignored).
- `https://huggingface.co/api/datasets/wangzhang/abliterix-datasets` returns **HTTP 401**
  without a token, so the dataset is private, hidden, or missing (public gated datasets
  return 200). Upstream's `docs/runpod-quickstart.md` mentions the error "private
  dataset not on the pod". **Unverified:** access, folder layout (`<subset>/*.json`
  assumed), the 1000-row file names, and the license.
- Whether `harmful_1000` contains all of `harmful_500` (and the same for `good`)
  **could not be checked**. `setup_data.py` prints only counts and yes/no answers.
  `evaluate.py` computes the unseen set at run time and stops if it is empty.

## Heretic (separate install, not in the project lock)

- **heretic-llm 1.4.0** (latest), Python 3.12. Its own install brings torch 2.15.0,
  transformers 5.22.0, accelerate 1.15.0, peft 0.21.2 and lm-eval 0.4.13.
- `heretic --help`: `--model` (Hub ID or local path) and `--evaluate-model` (model to
  compare against `--model`). The README documents
  `heretic --model <base> --evaluate-model <modified>`.
- Local paths go straight to `from_pretrained`. Evaluate mode does not contact the Hub
  except to download models. **Unverified** with real folders on a GPU machine.
- Defaults (`config.default.toml`): harmless prompts `mlabonne/harmless_alpaca` and
  harmful prompts `mlabonne/harmful_behaviors`, 100 each, column `text`, 100-token
  answers, system prompt "You are a helpful assistant.", and it skips gpt-oss's
  reasoning step. A response counts as a refusal if it is empty or contains a refusal
  keyword (case-insensitive). KL divergence is measured on the first generated token
  over the 100 harmless prompts. Heretic reads `config.toml` from its working
  directory, so the script runs it inside `runs/heretic_crosscheck/`.
- Output lines: a banner ending `v1.4.0` (confirmed by running it),
  `Loading good evaluation prompts from X...`, `* Initial refusals: N/M` (base), then
  `* KL divergence: 0.1234` and `* Refusals: N/M` (modified). Colors are turned off
  (`NO_COLOR`, plus stripping) and `COLUMNS=1000` stops line wrapping. Only version
  1.4.0 output is parsed.
- Memory: it evaluates the base model, frees it, then loads the modified model, so the
  peak is about one model. The original model may stay 4-bit (~65 GB) or be expanded
  to BF16 (~234 GB) depending on GPU support (**unverified**). The modified model is
  BF16 (~234 GB), spread across all visible GPUs.
- PyTorch: loading 4-bit gpt-oss needs 2.6 or newer. Heretic's install gets 2.15.0;
  the main lock has 2.11.0.
- Installing abliterix and Heretic together **works** (no conflicts). They are kept
  separate anyway so Heretic's newer packages don't change the abliterix environment.
