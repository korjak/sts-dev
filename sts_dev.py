from typing import Callable, Dict
import numpy as np
from scipy.stats import pearsonr
import csv
import torch
import os
import re


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

#aint much but it's honest work
#returns Pearson correlation coefficient, but p-value is also calculated
def compute_pearson(x, y):
    return pearsonr(x, y)[0]

def eval_hf(model_name: str, set_types: str = "all", model_dictionary: str = "./hf_models"):

    data_args = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/STS-B")
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=1,
        finetuning_task="sts-b",
    )

    train_args = TrainingArguments(
        do_eval=True,
        output_dir=f"./hf_eval/{model_name}",
    )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return glue_compute_metrics(
                task_name, np.squeeze(p.predictions), p.label_ids
            )

        return compute_metrics_fn

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    train_test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="train")
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="test")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=build_compute_metrics_fn("sts-b"),
    )

    return {"train": trainer.evaluate(eval_dataset=train_test_dataset), "dev": trainer.evaluate(eval_dataset=eval_dataset)}

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def train_hf(
    model_name: str,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
    from_checkpoint=False, checkpoint_path="./hf_models/",
) -> Dict[str, float]:
    if data_args is None:
        data_args = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/STS-B")
    if config is None:
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=1,
            finetuning_task="sts-b",
        )
    if train_args is None:
        train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            num_train_epochs=5.0,
            output_dir=f"./hf_models/{model_name}",
        )

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return glue_compute_metrics(
                task_name, np.squeeze(p.predictions), p.label_ids
            )

        return compute_metrics_fn

    #here we still need model_name to be solely a name, not a path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

    #with this condition we can change model_name to the path to the latest checkpoint
    if from_checkpoint==True:
        #we search for every checkpoint created for our model
        checkpoints=next(os.walk(checkpoint_path+model_name))[1]
        #and set model_name as the path to the one with highest number
        model_name=checkpoint_path+model_name+"/"+str(max(checkpoints,key=extract_number))

    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        compute_metrics=build_compute_metrics_fn("sts-b"),
    )

    trainer.train(model_path=model_name)
    trainer.save_model()

    return trainer.evaluate(eval_dataset=eval_dataset)
