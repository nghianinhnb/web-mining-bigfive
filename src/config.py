from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

PAN15_TRAIN_DIR = RAW_DIR / "pan15_train"
PAN15_TEST_DIR = RAW_DIR / "pan15_test"
PAN15_TRAIN_EN_DIR = RAW_DIR / "pan15_train_en"

N_TWEETS_PER_USER = 200
MAX_TOKENS_PER_TWEET = 64
TEST_SIZE = 0.2
DEV_SIZE = 0.1
SEED = 42

TRAIT_NAMES = ["open", "conscientious", "extroverted", "agreeable", "stable"]
TRAIT_COLS = [f"y_{t}" for t in TRAIT_NAMES]

# Languages
LANGUAGES = ["en", "es", "it", "nl"]

# Models
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "cardiffnlp/twitter-roberta-base-emotion"
ENCODER_MODEL = "cardiffnlp/twitter-roberta-base"
MULTILINGUAL_ENCODER_MODEL = "xlm-roberta-base"  # or "cardiffnlp/twitter-xlm-roberta-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHROMA_DIR = PROCESSED_DIR / "chroma_db"
BM25_INDEX_PATH = PROCESSED_DIR / "ir_bm25.pkl"
EVIDENCE_PATH = PROCESSED_DIR / "evidence_topk.parquet"

TOP_K_EVIDENCE = 5
TOP_K_RECS = 10

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

