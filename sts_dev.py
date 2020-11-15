from typing import Callable, Dict, List
import numpy as np
from scipy.stats import pearsonr
import csv
import torch

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
    InputExample
)

def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        return glue_compute_metrics(
            task_name, np.squeeze(p.predictions), p.label_ids
        )

    return compute_metrics_fn

def train_hf(
    model_name: str,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
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
            max_steps=1200,
            warmup_steps=120,
            output_dir="./hf_models/",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

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

# def evaluate(model_name: str) -> Dict[str, Dict[str, float]]:
#     data_args_only_sts = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/STS-B")
#     data_args_with_mrpc = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/combined")
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if model_name == "gpt2":
#         tokenizer.pad_token = "[PAD]"
#     eval_dataset_only_sts = GlueDataset(data_args_only_sts, tokenizer=tokenizer, mode="dev")
#     eval_dataset_with_mrpc = GlueDataset(data_args_with_mrpc, tokenizer=tokenizer, mode="dev")
#     config = AutoConfig.from_pretrained(
#             model_name,
#             num_labels=1,
#             finetuning_task="sts-b",
#         )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         model_name,
#         config=config,
#     )
#     train_args = TrainingArguments(
#             do_train=False,
#             do_eval=True,
#             output_dir="./eval_only/",
#         )
#     trainer = Trainer(
#         model=model,
#         args=train_args,
#         compute_metrics=build_compute_metrics_fn("sts-b"),
#     )
#     return {"sts_only": trainer.evaluate(eval_dataset=eval_dataset_only_sts), "with_mrpc": trainer.evaluate(eval_dataset=eval_dataset_with_mrpc)}

def evaluate(model_name: str) -> int:
    dev_sts = []
    dev_combined = []
    for set_type, set_dir in zip([dev_sts, dev_combined], ["STS-B", "combined"]):
        with open(f"./data/{set_dir}/dev.tsv") as f:
            ftsv = csv.reader(f, delimiter="\t")
            next(ftsv, None)
            for l in ftsv:
                set_type.append((l[0], l[7], l[8], float(l[9])))
        break
    # dev_sts_features = glue_convert_examples_to_features(list(map(lambda ex: ex[1], dev_sts)))
    # dev_combined_features = glue_convert_examples_to_features(list(map(lambda ex: ex[1], dev_combined)))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    features_sts = tokenizer(list(map(lambda ex: (ex[1], ex[2]), dev_sts)), padding='max_length', truncation=True, max_length=500)
    # features_combined = tokenizer(list(map(lambda ex: (ex[1], ex[2]), dev_combined)), padding='max_length', truncation=True, max_length=500)
    results = model(torch.tensor(features_sts['input_ids']).cuda(), attention_mask=torch.tensor(features_sts["attention_mask"]).cuda())
    print(results)


# def evaluate_hf(model_name: str, sets: List[List[InputFeature]]) -> int:
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = TFAutoModel.from_pretrained(model_name)
#     for set_type in sets:
#         predictions = model(set_type)

# def evaluate_st(model_name: str) -> int:
#     pass