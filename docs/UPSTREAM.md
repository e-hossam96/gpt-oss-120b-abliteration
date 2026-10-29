# Upstream sources

## abliterix

- Repository: <https://github.com/wuwangzhang1216/abliterix> (AGPL-3.0)
- Commit: `1940c8c14688772011d5f20b03c5dffb042a3e8f` (master on 2026-10-29, "fix: resolve
  baseline restore idempotency, resource leaks, and eval metrics (#107)", 2026-09-28).
- Installed package: `abliterix==1.12.2` from PyPI (tag `v1.12.2` =
  `a0a91f09472e1db019c1ca37a6de568fa089e930`, 27 commits behind master).

Files copied unchanged from that commit:

| Local path | Upstream path | SHA256 |
| --- | --- | --- |
| `configs/upstream/default.toml` | `configs/default.toml` | `4f93cfc8333f8ec30d7986895da2ea24e51d60f00c30b36d05117b99adbfd63f` |
| `configs/upstream/gpt_oss_120b_v7s1.toml` | `configs/gpt_oss_120b_v7s1.toml` | `e853e50195bd953b97c3360daf5185749f5253786c26a54dc33de748d4463bd9` |
| `configs/upstream/gpt_oss_120b_v7s2.toml` | `configs/gpt_oss_120b_v7s2.toml` | `4094e9ad10896452eaef4783f7d96b211451a011a907b8b44e7d21984a6911c0` |
| `scripts/upstream/prepare_bf16_checkpoint.py` | `scripts/prepare_bf16_checkpoint.py` | `b95e56999d96a2b2d80a71ad6755394715edf290c9862d1e4346440e15ddc38e` |
| `scripts/upstream/export_model.py` | `scripts/export_model.py` | `ec3d4d02585c45c0936105a700ec8c200449803a28503cdb6f7396d79e706a2e` |

The two scripts are in the upstream repo but **not** in the PyPI package. Both were
last changed before v1.12.2, and every abliterix function they call exists in 1.12.2.
ruff skips them so they stay unchanged.

Also read at the same commit (not copied):

- `benchmarks/SPEC.md`: checksums and fields of `harmful_500/harmful_prompts_500.json`
  and `good_500/good_prompts_500.json`, used by `scripts/setup_data.py`.
- `quick_start/deploy_gpt_oss_120b_v7.sh`: upstream's two-stage pipeline (BF16
  conversion → Stage 1 → export → Stage 2 → export), which this repo follows.

## Heretic (optional)

- Repository: <https://github.com/p-e-w/heretic> (AGPL-3.0), PyPI `heretic-llm`.
- Version: `1.4.0` (latest on 2026-10-29). Source read from tag `v1.4.0`; master was
  `efb1be6e3f979581b8878be978e001e34260159b`.
- Not a dependency and nothing copied. `scripts/heretic_crosscheck.py` runs it as a
  separate program.

## Model

- `openai/gpt-oss-120b`, commit `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, Apache-2.0.
  36 layers, 64 attention heads, 8 KV heads, 128 experts (4 active per token), stored
  in MXFP4 (4-bit) format.
