# from huggingface_hub import snapshot_download
# import os

# os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"
# os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"  # adjust to CPU cores
# os.environ["HF_ENABLE_PARALLEL_LOADING"] = True
# # export HF_ENDPOINT=https://hf-mirror.com

# snapshot_download(
#     repo_id="allenai/olmOCR-2-7B-1025",
#     local_dir="./olmocr_model",
#     local_dir_use_symlinks=False,
#     resume_download=True,
#     max_workers=8
# )

# print("✅ Download complete or resumed!")

import subprocess
from huggingface_hub import HfApi

repo_id = "allenai/olmOCR-2-7B-1025"

api = HfApi()

files = api.list_repo_files(repo_id)

# Filter large useful files
files = [f for f in files if not f.endswith(".md")]

download_dir = "./olmocr_model"

for file in files:
    url = f"https://huggingface.co/{repo_id}/resolve/main/{file}"
    
    cmd = [
        "aria2c",
        "-x", "16",                # 16 connections per file
        "-s", "16",                # 16 parallel segments
        "-k", "1M",                # 1MB chunks
        "--continue=true",         # resume
        "--auto-file-renaming=false",
        "-d", download_dir,
        "-o", file.replace("/", "_"),
        url
    ]
    
    subprocess.run(cmd)
