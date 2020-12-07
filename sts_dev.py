from typing import Callable, Dict, Tuple, List
import numpy as np
from scipy.stats import pearsonr
import csv
import torch
import os
import re
import time
import traceback
import pandas as pd
# from sentence_transformers import SentenceTransformer, util

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
    GlueDataset,
    GlueDataTrainingArguments,
    TrainingArguments,
    EvalPrediction,
    glue_compute_metrics,
)

# aint much but it's honest work
# returns Pearson correlation coefficient, but p-value is also calculated
def compute_pearson(x, y) -> float:
    return pearsonr(x, y)[0]


def extract_number(f):
    s = re.findall("\d+$", f)
    return (int(s[0]) if s else -1, f)


def get_classes(
    model_type: str,
) -> Tuple[PretrainedConfig, PreTrainedTokenizer, PreTrainedModel]:
    if model_type == "auto":
        return AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
    else:
        raise Exception("Model type not recognized, currently supported modes: [auto]")


def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        return glue_compute_metrics(task_name, np.squeeze(p.predictions), p.label_ids)

    return compute_metrics_fn


def rescale_cosine_scale(cosine_scores: List[float]) -> List[float]:
    # cosine_scores = list(map(lambda val: val - min(cosine_scores), cosine_scores))
    # cosine_scores = list(map(lambda val: val / max(cosine_scores), cosine_scores))
    cosine_scores = list(map(lambda val: 0 if val < 0 else val, cosine_scores))
    cosine_scores = list(map(lambda val: val * 5, cosine_scores))
    return cosine_scores


def load_set(set_type: str = "dev", set_directory: str = "./data/STS-B") -> List[float]:
    with open(os.path.join(set_directory, f"{set_type}.tsv")) as f:
        fcsv = csv.reader(f, delimiter="\t")
        next(fcsv)
        return [float(l[-1]) for l in fcsv]


def eval_sbert(model_name: str, dataset_dir: str = "./data/STS-B"):
    datasets = {
        "train": os.path.join(dataset_dir, "train.tsv"),
        "dev": os.path.join(dataset_dir, "dev.tsv"),
    }
    results = {}
    for set_type, dataset in datasets.items():
        all_sentences = pd.read_csv(
            dataset,
            delimiter="\t",
        )
        model = SentenceTransformer(model_name)
        embeddings1 = model.encode(all_sentences["sentence1"], convert_to_tensor=True)
        embeddings2 = model.encode(all_sentences["sentence2"], convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
        cosine_scores = [
            float(cosine_scores[i][i]) for i in range(len(all_sentences.index))
        ]
        cosine_scores = rescale_cosine_scale(cosine_scores)
        gold_scores = load_set(set_type)
        results[set_type] = {
            "eval_pearson": compute_pearson(cosine_scores, gold_scores)
        }
    return results


def eval_hf(
    model_name: str,
    set_types: str = "all",
    from_path: bool = True,
    checkpoint_path: str = "./hf_models",
    dataset_type: str = "STS-B",
    model_type: str = "auto",
    fixed_checkpoint_no: int = None,
):
    config_class, tokenizer_class, model_class = get_classes(model_type)
    model_directory = model_name.replace("/", "_")
    data_args = GlueDataTrainingArguments(
        task_name="sts-b", data_dir=f"./data/{dataset_type}"
    )
    config = config_class.from_pretrained(
        model_name,
        num_labels=1,
        finetuning_task="sts-b",
    )

    train_args = TrainingArguments(do_eval=True, output_dir="/data/out")

    # print(os.listdir(os.path.join(checkpoint_path, model_directory)))
    if from_path:
        checkpoints = next(os.walk(os.path.join(checkpoint_path, model_directory)))[1]
        if fixed_checkpoint_no is not None:
            checkpoint_no = fixed_checkpoint_no
        else:
            checkpoint_no = max(checkpoints, key=extract_number)
        model_path = os.path.join(
            checkpoint_path, model_directory, "checkpoint-" + str(checkpoint_no)
        )
    else:
        model_path = model_name

    tokenizer = tokenizer_class.from_pretrained(model_name)
    train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
    eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

    model = model_class.from_pretrained(
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

def predict_hf(
    model_name: str,
    set_types: str = "all",
    from_path: bool = True,
    checkpoint_path: str = "./hf_models",
    dataset_type: str = "STS-B",
    model_type: str = "auto",
    fixed_checkpoint_no: int = None,
):
    config_class, tokenizer_class, model_class = get_classes(model_type)
    model_directory = model_name.replace("/", "_")
    data_args = GlueDataTrainingArguments(
        task_name="sts-b", data_dir=f"./data/{dataset_type}"
    )
    config = config_class.from_pretrained(
        model_name,
        num_labels=1,
        finetuning_task="sts-b",
    )

    train_args = TrainingArguments(do_eval=True, output_dir="/data/out")

    # print(os.listdir(os.path.join(checkpoint_path, model_directory)))
    if from_path:
        checkpoints = next(os.walk(os.path.join(checkpoint_path, model_directory)))[1]
        if fixed_checkpoint_no is not None:
            checkpoint_no = fixed_checkpoint_no
        else:
            checkpoint_no = extract_number(max(checkpoints, key=extract_number))[0]
        model_path = os.path.join(
            checkpoint_path, model_directory, "checkpoint-" + str(checkpoint_no)
        )
    else:
        model_path = model_name

    tokenizer = tokenizer_class.from_pretrained(model_name)
    test_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="test")

    model = model_class.from_pretrained(
        model_path,
        config=config,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=build_compute_metrics_fn("sts-b"),
    )

    return trainer.predict(test_dataset=test_dataset)

def train_hf(
    model_name: str,
    output_dir: str = None,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
    model_type: str = "auto",
    dataset_type: str = "STS-B",
    from_checkpoint: bool = False,
    checkpoint_path="./hf_models/",
) -> Dict[str, float]:
    name_only = model_name
    model_directory = model_name.replace("/", "_")
    try:
        tic = time.time()
        config_class, tokenizer_class, model_class = get_classes(model_type)
        if data_args is None:
            data_args = GlueDataTrainingArguments(
                task_name="sts-b", data_dir=f"./data/{dataset_type}"
            )
        if config is None:
            config = config_class.from_pretrained(
                model_name,
                num_labels=1,
                finetuning_task="sts-b",
            )
        if train_args is None:
            assert (
                output_dir is not None
            ), "Either train_args or output_dir must be provided"
            train_args = TrainingArguments(
                evaluate_during_training=True,
                do_train=True,
                do_eval=True,
                num_train_epochs=5.0,
                output_dir=output_dir,
            )

        # here we still need model_name to be solely a name, not a path
        tokenizer = tokenizer_class.from_pretrained(model_name)
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        # with this condition we can change model_name to the path to the latest checkpoint
        if from_checkpoint == True:
            # we search for every checkpoint created for our model
            checkpoints = next(os.walk(os.path.join(checkpoint_path, model_directory)))[
                1
            ]
            # and set model_name as the path to the one with highest number
            model_name = os.path.join(
                checkpoint_path,
                model_directory,
                str(max(checkpoints, key=extract_number)),
            )

        model = model_class.from_pretrained(
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
        return {
            "model_name": name_only,
            "exception": str(exe),
            "stacktrace": traceback.print_exc(),
        }

def hyp_tuning_hf(
    model_name: str,
    output_dir: str = None,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
    model_type: str = "auto",
    dataset_type: str = "STS-B",
    from_checkpoint: bool = False,
    checkpoint_path="./hf_models/",
) -> Dict[str, float]:
    name_only = model_name
    model_directory = model_name.replace("/", "_")
    try:
        tic = time.time()
        config_class, tokenizer_class, model_class = get_classes(model_type)
        if data_args is None:
            data_args = GlueDataTrainingArguments(
                task_name="sts-b", data_dir=f"./data/{dataset_type}"
            )
        if config is None:
            config = config_class.from_pretrained(
                model_name,
                num_labels=1,
                finetuning_task="sts-b",
            )
        if train_args is None:
            assert (
                output_dir is not None
            ), "Either train_args or output_dir must be provided"
            train_args = TrainingArguments(
                evaluate_during_training=True,
                do_train=True,
                do_eval=True,
                output_dir=output_dir,
            )

        # here we still need model_name to be solely a name, not a path
        tokenizer = tokenizer_class.from_pretrained(model_name)
        train_dataset = GlueDataset(data_args, tokenizer=tokenizer)
        eval_dataset = GlueDataset(data_args, tokenizer=tokenizer, mode="dev")

        # with this condition we can change model_name to the path to the latest checkpoint
        if from_checkpoint == True:
            # we search for every checkpoint created for our model
            checkpoints = next(os.walk(os.path.join(checkpoint_path, model_directory)))[
                1
            ]
            # and set model_name as the path to the one with highest number
            model_name = os.path.join(
                checkpoint_path,
                model_directory,
                str(max(checkpoints, key=extract_number)),
            )

        def model_init():
            return AutoModelForSequenceClassification.from_pretrained(name_only, return_dict=True)

        trainer = Trainer(
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=build_compute_metrics_fn("sts-b"),
        )

        trainer.train(model_path=model_name)
        trainer.save_model()
        t = time.time() - tic

        return {
            "model_name": name_only,
            "parameters": trainer.hyperparameter_search(
                direction="maximize",
                n_samples=10
            )
        }
    except Exception as exe:
        return {
            "model_name": name_only,
            "exception": str(exe),
            "stacktrace": traceback.print_exc(),
        }
