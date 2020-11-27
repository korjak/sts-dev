import csv
import json
import os

DATASET_TYPE = "STS-B"

RESULTS_DIRECTORY = f"/data/hf_eval/combined/not_parsed"
PARSED_DIRECTORY = f"/data/hf_eval/{DATASET_TYPE}/parsed"
with open(os.path.join(PARSED_DIRECTORY, "parsed_all_trained_on_combined.tsv"), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    for model_res in os.scandir(RESULTS_DIRECTORY):
        with open(model_res.path) as f:
            fj = json.load(f)
            writer.writerow([fj["model_name"], *list(map(lambda kv: str(kv[1]["train"]["eval_pearson"]), fj["checkpoints"].items()))])
            writer.writerow([fj["model_name"], *list(map(lambda kv: str(kv[1]["dev"]["eval_pearson"]), fj["checkpoints"].items()))])
