from sts_dev import eval_hf
from models import directory_to_model_name, siamese_models
import os
import json

DATASET_TYPE_MODEL = "combined"
DATASET_TYPE_DATASET = "STS-B"

MODEL_DIRECTORY = f"/data/hf_models/{DATASET_TYPE_MODEL}"
RESULTS_DIRECTORY = f"/data/hf_eval/{DATASET_TYPE_DATASET}/not_parsed"

for model_dir in os.scandir(MODEL_DIRECTORY):
    checkpoints = []
    evaluation = {}
    for checkpoint_dir in os.scandir(model_dir):
        if checkpoint_dir.name.startswith("checkpoint"):
            checkpoints.append(int(checkpoint_dir.name[11:]))
    for ch in checkpoints:
        evaluation[str(ch)] = eval_hf(model_name=directory_to_model_name[model_dir.name], checkpoint_path=MODEL_DIRECTORY, fixed_checkpoint_no=ch, dataset_type=DATASET_TYPE_DATASET) 
    with open(os.path.join(RESULTS_DIRECTORY, f"{model_dir.name}.json"), "w") as f:
        json.dump({"model_name": directory_to_model_name[model_dir.name], "checkpoints": evaluation}, f)
