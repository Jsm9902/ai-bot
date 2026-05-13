import os

# 환경 설정
os.environ["ANONYMIZED_TELEMETRY"] = "False"

MODEL_NAME = "hf.co/MLP-KTLim/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M"
UPLOAD_DIR = "data"
DB_PATH = "chroma_db"
MAX_STORAGE_MB = 200

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)