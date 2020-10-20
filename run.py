from sts_dev import train
import json

all_models = [
    "SparkBeyond/roberta-large-sts-b",
    "bert-base-uncased",
    "gpt2",
    "albert-base-v2",
    "xlnet-large-cased",
]

results = list(map(train, all_models))
print(results)
with open("results.json", "w") as f:
    json.dump(results, f)
# print(f"Best model: {max(evals.items(), lambda kv: kv[1])}")
