import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import hf_hub_download

TARGETS = [
    ("unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
     "Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
     "/home/mateo/local-llm/models/llama-4-scout"),
    ("unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
     "Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf",
     "/home/mateo/local-llm/models/llama-4-scout"),
    ("unsloth/Qwen3-Coder-Next-GGUF",
     "Qwen3-Coder-Next-UD-Q4_K_M.gguf",
     "/home/mateo/local-llm/models/qwen3-coder-next"),
]

for repo, filename, local_dir in TARGETS:
    print(f"=== downloading {filename} ===", flush=True)
    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir)
    print(f"=== done: {path} ===", flush=True)

print("ALL DOWNLOADS COMPLETE", flush=True)
