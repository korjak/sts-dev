from sts_dev import train_hf
import json
from datetime import datetime
import os
from models import all_models
from transformers import TrainingArguments

DATASET_TYPE = "STS-B"
EXPERIMENT_NO = 24
# os.mkdir(f"/data/hf_models/{DATASET_TYPE}/experiment_{str(EXPERIMENT_NO)}")
# os.mkdir(f"/data/hf_results/{DATASET_TYPE}/experiment_{str(EXPERIMENT_NO)}")
for model_name in all_models:
    # os.mkdir(f"/data/hf_models/{model_name.replace('/', '_')}")
    try:
        result = train_hf(
            model_name,
            dataset_type=DATASET_TYPE,
            from_checkpoint=False,
            # checkpoint_path=f"/data/hf_models/{DATASET_TYPE}/experiment_org",
            train_args=TrainingArguments(
                do_train=True,
                # do_eval=True,
                # adam_epsilon=1e-5,
                learning_rate=5e-3,
                num_train_epochs=3.0,
                # weight_decay=0.5,
                output_dir=f"/data/hf_models/{DATASET_TYPE}/experiment_{str(EXPERIMENT_NO)}/{model_name.replace('/', '_')}",
            ),
        )
        with open(
            f"/data/hf_results/{DATASET_TYPE}/experiment_{str(EXPERIMENT_NO)}/result_{model_name.replace('/', '_')}.json",
            "w",
        ) as f:
            json.dump(result, f)
    except Exception as exe:
        print(exe)
