from typing import Callable, Dict
import numpy as np
from scipy.stats import pearsonr
import csv
import torch
import os
import re
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    AutoConfig,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)

# aint much but it's honest work
# returns Pearson correlation coefficient, but p-value is also calculated
def compute_pearson(x, y):
    return pearsonr(x, y)[0]


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def eval_hf(
    model_name: str,
    set_types: str = "all",
    checkpoint_path: str = "./hf_models",
    fixed_checkpoint_no: int = None,
):
    model_directory = model_name.replace("/", "_")
    data_args = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/STS-B")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=1,
        finetuning_task="sts-b",
    )

    train_args = TrainingArguments(
        do_eval=True,
        output_dir="/data/out"
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return glue_compute_metrics(
                task_name, np.squeeze(p.predictions), p.label_ids
            )

        return compute_metrics_fn
    # print(os.listdir(os.path.join(checkpoint_path, model_directory)))
    checkpoints = next(os.walk(os.path.join(checkpoint_path, model_directory)))[1]
    if fixed_checkpoint_no is not None:
        checkpoint_no = fixed_checkpoint_no
    else:
        checkpoint_no = max(checkpoints, key=extract_number)
    model_path = os.path.join(checkpoint_path, model_directory,  "checkpoint-" + str(checkpoint_no))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=build_compute_metrics_fn("sts-b"),
    )

    return {
        "train": trainer.evaluate(eval_dataset=train_dataset),
        "dev": trainer.evaluate(eval_dataset=eval_dataset),
    }


def train_hf(
    model_name: str,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
    from_checkpoint=False,
    checkpoint_path="./hf_models/",
) -> Dict[str, float]:
    name_only = model_name
    try:
        tic = time.time()
        if data_args is None:
            data_args = GlueDataTrainingArguments(
                task_name="sts-b", data_dir="./data/STS-B"
            )
        if config is None:
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=1,
                finetuning_task="sts-b",
            )
        if train_args is None:
            train_args = TrainingArguments(
                evaluate_during_training=True,
                do_train=True,
                do_eval=True,
                num_train_epochs=5.0,
                output_dir=f"/data/hf_models/{model_name.replace('/', '_')}",
            )

        def build_compute_metrics_fn(
            task_name: str,
        ) -> Callable[[EvalPrediction], Dict]:
            def compute_metrics_fn(p: EvalPrediction):
                return glue_compute_metrics(
                    task_name, np.squeeze(p.predictions), p.label_ids
                )

            return compute_metrics_fn

        # here we still need model_name to be solely a name, not a path
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        # with this condition we can change model_name to the path to the latest checkpoint
        if from_checkpoint == True:
            # we search for every checkpoint created for our model
            checkpoints = next(os.walk(checkpoint_path + model_name))[1]
            # and set model_name as the path to the one with highest number
            model_name = (
                checkpoint_path
                + model_name
                + "/"
                + str(max(checkpoints, key=extract_number))
            )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=build_compute_metrics_fn("sts-b"),
        )

        trainer.train(model_path=model_name)
        trainer.save_model()
        t = time.time() - tic

        return {
            "model_name": name_only,
            "train": trainer.evaluate(eval_dataset=train_dataset),
            "dev": trainer.evaluate(eval_dataset=eval_dataset),
            "time": t,
        }
    except Exception as exe:
        return {"model_name": name_only, "exception": str(exe)}
