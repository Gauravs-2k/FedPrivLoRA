import os
from pathlib import Path
from huggingface_hub import HfApi, upload_folder

from app.utils.config import settings

hf_token = settings.HF_TOKEN or os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not set. Please set your Hugging Face token in settings or environment.")

api = HfApi(token=hf_token)

base_repo = "Gaurav2k/qwen1.5-1.8b-chat"

departments = {
    "engineering": "qwen_dept_lora_engineering",
    "finance": "qwen_dept_lora_finance",
    "hr": "qwen_dept_lora_hr",
    "it_support": "qwen_dept_lora_it_support",
}

for dept, local_dir in departments.items():
    repo_name = f"{base_repo}-{dept}"
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"Directory {local_path} does not exist. Skipping {dept}.")
        continue
    
    try:
        api.create_repo(repo_name, private=False, exist_ok=True)
        print(f"Repo {repo_name} created or already exists.")
    except Exception as e:
        print(f"Error creating repo {repo_name}: {e}")
        continue
    
    try:
        upload_folder(
            folder_path=str(local_path),
            repo_id=repo_name,
            token=hf_token,
            commit_message=f"Upload LoRA adapter for {dept} department",
        )
        print(f"Successfully uploaded {dept} to {repo_name}")
    except Exception as e:
        print(f"Error uploading {dept}: {e}")

print("All uploads completed.")