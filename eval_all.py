from sts_dev import evaluate
from models import list_all
import json

# evals = {}
# for mod in list_all:
#     try:
#         results = evaluate(mod)
#     except Exception as ex:
#         print(ex)
#         continue
#     print(f"{mod}: {results}")
#     evals[mod] = results
# json.dump(evals, open("./results/without_training.json", "w"))

evaluate("SparkBeyond/roberta-large-sts-b")