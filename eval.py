from sts_dev import eval_hf
from models import directory_to_model_name
import os
import json

MODEL_DIRECTORY = "/data/hf_models"
RESULTS_DIRECTORY = "/data/hf_eval/not_parsed"
for model_dir in os.scandir(MODEL_DIRECTORY):
    checkpoints = []
    evaluation = {}
    for checkpoint_dir in os.scandir(model_dir):
        if checkpoint_dir.name.startswith("checkpoint"):
            checkpoints.append(int(checkpoint_dir.name[11:]))
    for ch in checkpoints:
        evaluation[str(ch)] = eval_hf(model_name=directory_to_model_name[model_dir], model_dictionary=MODEL_DICTIONARY, fixed_checkpoint_no=ch)
    with open(os.path.join(RESULTS_DIRECTORY, f"{model_dir}.json"), "w") as f:
        json.dump(evaluation, f)


