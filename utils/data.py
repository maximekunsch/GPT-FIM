from datasets import load_dataset

ds = load_dataset("the-stack-smol", split="train")  # tiny Python subset
print(ds[0])
