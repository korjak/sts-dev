from sts_dev import train_hf
import json
from datetime import datetime
import os
from models import all_models

DATASET_TYPE = "combined"

for model_name in all_models:
    # os.mkdir(f"/data/hf_models/{model_name.replace('/', '_')}")
    try:
        result = train_hf(model_name, dataset_type=DATASET_TYPE)
        with open(f"/data/hf_results/{DATASET_TYPE}/result_{model_name.replace('/', '_')}.json", "w") as f:
            json.dump(result, f)
    except Exception as exe:
        print(exe)

