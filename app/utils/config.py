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


settings = Config()