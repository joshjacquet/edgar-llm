import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output")

# Model Choices
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL_NAME = "google/flan-t5-base"

# Target variables info
TARGET_VARIABLES = {
    "numeric": ["Total Revenues", "Net Income"],
    "categorical": ["State of Incorporation"],
}
