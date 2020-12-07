import csv
import json
import os

EXPERIMENT_NO = 24

RESULTS_DIRECTORY = f"/data/hf_eval/STS-B/experiment_{str(EXPERIMENT_NO)}/not_parsed"
PARSED_DIRECTORY = f"/data/hf_eval/STS-B/experiment_{str(EXPERIMENT_NO)}/parsed"
with open(os.path.join(PARSED_DIRECTORY, "parsed_all.tsv"), "w") as f:
    writer = csv.writer(f, delimiter="\t")
    for model_res in os.scandir(RESULTS_DIRECTORY):
        with open(model_res.path) as f:
            fj = json.load(f)
            writer.writerow(
                [
                    fj["model_name"],
                    *list(
                        map(
                            lambda kv: str(kv[1]["train"]["eval_pearson"]),
                            fj["checkpoints"].items(),
                        )
                    ),
                ]
            )
            writer.writerow(
                [
                    fj["model_name"],
                    *list(
                        map(
                            lambda kv: str(kv[1]["dev"]["eval_pearson"]),
                            fj["checkpoints"].items(),
                        )
                    ),
                ]
            )
