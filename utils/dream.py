from datasets import load_dataset
import os

ds = load_dataset(
    "bigcode/the-stack-v2-dedup",
    "Python",
    split="train",
    token=os.environ["HF_TOKEN"],
    cache_dir="C:/Users/maxim/.cache/huggingface/datasets",
    download_mode="reuse_dataset_if_exists"
)


# Save to a plain text file, one example per line
with open("the-stack.txt", "w", encoding="utf-8") as f:
    for example in ds:
        f.write(example["content"] + "\n")  # replace "content" if your column is named differently