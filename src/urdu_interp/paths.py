from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"

POETRY_DIR = DATA_DIR / "poetry"
FLORES_DIR = DATA_DIR / "parallel" / "flores200"
OPUS_UR_EN_DIR = DATA_DIR / "parallel" / "opus_ur_en"
OPUS_HI_EN_DIR = DATA_DIR / "parallel" / "opus_hi_en"
OPUS_FA_EN_DIR = DATA_DIR / "parallel" / "opus_fa_en"
EBIBLE_DIR = DATA_DIR / "religious" / "ebible"
TANZIL_DIR = DATA_DIR / "religious" / "tanzil"
TRANSLIT_DIR = DATA_DIR / "translit"

MODELS_YAML = CONFIGS_DIR / "models.yaml"

DATA_CACHE_DIR = RESULTS_DIR / "data_cache"
RQ1_DIR = RESULTS_DIR / "rq1_convergence"
RQ2_DIR = RESULTS_DIR / "rq2_logit_lens"
RQ3_DIR = RESULTS_DIR / "rq3_finetune"
FINETUNED_MODELS_DIR = RESULTS_DIR / "finetuned_models"


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, DATA_CACHE_DIR, RQ1_DIR, RQ2_DIR, RQ3_DIR, FINETUNED_MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
