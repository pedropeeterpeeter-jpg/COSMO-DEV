from pathlib import Path

# =====================================================
# DIRECTÓRIOS BASE
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "relatorios"

# ─────────────────────────────────────────────────────
# CORRECÇÃO: IMAGE_DIR aponta agora para static/imagens
# que é onde o Flask procura ficheiros estáticos.
# O caminho anterior (BASE_DIR / "imagens") guardava as
# imagens fora da pasta static — o HTML não as encontrava.
# ─────────────────────────────────────────────────────
IMAGE_DIR = BASE_DIR / "static" / "imagens"

# Cria as pastas automaticamente se não existirem
MODEL_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# DADOS
# =====================================================

RAW_DATA = (
    DATA_DIR
    / "raw"
    / "Dataset_Completo_Abandono_Escolar_Angola_345.xlsx"
)

# =====================================================
# MODELOS
# ─────────────────────────────────────────────────────
# CORRECÇÃO: RANDOM_FOREST_PATH estava declarado duas
# vezes. Removida a duplicação.
# =====================================================

RANDOM_FOREST_PATH = MODEL_DIR / "random_forest.pkl"
SCALER_PATH        = MODEL_DIR / "scaler.pkl"

# =====================================================
# IMAGENS SHAP
# =====================================================

SHAP_GLOBAL = IMAGE_DIR / "shap_global.png"
SHAP_LOCAL  = IMAGE_DIR / "shap_local.png"

# =====================================================
# CONFIGURAÇÕES ML
# =====================================================

RANDOM_STATE  = 42
TEST_SIZE     = 0.20
TARGET_COLUMN = "Situacao_Escolar"
