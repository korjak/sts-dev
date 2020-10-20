from sts_dev import train
import json

all_models = [
    "SparkBeyond/roberta-large-sts-b",
    "bert-base-uncased",
    "gpt2",
    "albert-base-v2",
    "xlnet-large-cased",
]

for model in all_models:
    result = train(model)
    with open(f"result_{model}.json", "w") as f:
        json.dump(result, f)
# print(f"Best model: {max(evals.items(), lambda kv: kv[1])}")
