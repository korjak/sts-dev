from models import all_models, directory_to_model_name
from sts_dev import predict_hf
import csv
import os
import json
import pandas as pd
DATASET_TYPE_MODEL = "STS-B"
DATASET_TYPE_DATASET = "med_all"
EXPERIMENT_NO = 20
MODEL_DIRECTORY = (
    f"/data/hf_models/{DATASET_TYPE_MODEL}/experiment_{str(EXPERIMENT_NO)}"
)
TEST_FILE = f"./data/{DATASET_TYPE_DATASET}/test.tsv"
df = pd.read_csv(TEST_FILE, sep="\t")
# os.mkdir(f"/data/hf_predictions/{DATASET_TYPE_DATASET}/experiment_{str(EXPERIMENT_NO)}")
for model_dir in os.scandir(MODEL_DIRECTORY):
    if model_dir.name.startswith("distilbert"):
        # os.mkdir(f"/data/hf_models/{model_name.replace('/', '_')}")
        result = predict_hf(
                model_name=directory_to_model_name[model_dir.name],
                checkpoint_path=MODEL_DIRECTORY,
                dataset_type=DATASET_TYPE_DATASET,
            )
        # with open(
        #     f"/data/hf_predictions/{DATASET_TYPE_DATASET}/experiment_{str(EXPERIMENT_NO)}/result_{model_dir.name}.json",
        #     "w",
        # ) as f:
        #     json.dump(result, f)
        preds = list(map(lambda val: val[0], result.predictions))
        mdf = df.copy()
        # print(mdf.head)
        mdf["score"] = preds
        mdf.to_csv(f"./with_scores/all_med_distil_{model_dir.name}.tsv", sep="\t")