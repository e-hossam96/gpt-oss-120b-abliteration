from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
UPSTREAM_CONFIG_DIR = CONFIG_DIR / "upstream"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"

# Must match model_id in configs/stage*.toml.
BASE_BF16_DIR = "out/gpt-oss-120b-bf16"
STAGE1_EXPORT_DIR = "out/stage1"
STAGE2_EXPORT_DIR = "out/stage2"

STAGES = {
    "stage1": (CONFIG_DIR / "stage1.toml", UPSTREAM_CONFIG_DIR / "gpt_oss_120b_v7s1.toml"),
    "stage2": (CONFIG_DIR / "stage2.toml", UPSTREAM_CONFIG_DIR / "gpt_oss_120b_v7s2.toml"),
}
