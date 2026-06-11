import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "validation", "ground_truth.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "output")
PROMPTS_PATH = os.path.join(BASE_DIR, "prompts")

# Model Choices
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-large"
# LLM_MODEL_NAME = "google/flan-t5-base"

# Target variables info
TARGET_VARIABLES = {
    "numeric": ["Total Revenues", "Net Income"],
    "categorical": ["Stock Exchange"],
}

# Target 10-K
CIK = "5272"  # AIG
YEAR = "2020"
