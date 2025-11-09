from datasets import load_dataset

dataset = load_dataset("Vezora/Tested-143k-Python-Alpaca")

educational_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")

evol_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct")

mceval_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct")

package_instruct = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct")

# Save to a .txt file
dataset["train"].to_csv("python_alpaca_train.txt", index=False, sep="\n", header=False)

educational_instruct["train"].to_csv("educational_instruct.txt", index=False, sep="\n", header=False)

evol_instruct["train"].to_csv("evol_instruct.txt", index=False, sep="\n", header=False)

mceval_instruct["train"].to_csv("mceval_instruct.txt", index=False, sep="\n", header=False)

package_instruct["train"].to_csv("package_instruct.txt", index=False, sep="\n", header=False)