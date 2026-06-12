# download_qasper.py

from datasets import load_dataset
import os
import json

SAVE_DIR = "./data/qasper"
os.makedirs(SAVE_DIR, exist_ok=True)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

dataset = load_dataset(
    "allenai/qasper",
    trust_remote_code=True
)

print(dataset)

for split in dataset.keys():
    split_dir = os.path.join(SAVE_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, item in enumerate(dataset[split]):
        with open(
            os.path.join(split_dir, f"{i}.json"),
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

print("Download complete.")