import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for loading environment variables."""

    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.LLM_MODEL = os.getenv("LLM_MODEL")
        self.LLM_MODEL_ENDPOINT = os.getenv("LLM_MODEL_ENDPOINT")
        self.LLM_API_KEY = os.getenv("LLM_API_KEY")
        self.ADAPTER_REPOS = {
            "engineering": os.getenv("ADAPTER_REPO_ENGINEERING", "Gaurav2k/qwen2-0.5b-engineering"),
            "finance": os.getenv("ADAPTER_REPO_FINANCE", "Gaurav2k/qwen2-0.5b-finance"),
            "hr": os.getenv("ADAPTER_REPO_HR", "Gaurav2k/qwen2-0.5b-hr"),
            "it_support": os.getenv("ADAPTER_REPO_IT_SUPPORT", "Gaurav2k/qwen2-0.5b-it-support"),
        }
        self.ADAPTER_BASE_MODELS = {
            "engineering": os.getenv("ADAPTER_BASE_ENGINEERING", "Qwen/Qwen2-0.5B"),
            "finance": os.getenv("ADAPTER_BASE_FINANCE", "Qwen/Qwen2-0.5B"),
            "hr": os.getenv("ADAPTER_BASE_HR", "Qwen/Qwen2-0.5B"),
            "it_support": os.getenv("ADAPTER_BASE_IT_SUPPORT", "Qwen/Qwen2-0.5B"),
        }

settings = Config()