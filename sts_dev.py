from typing import Callable, Dict
import numpy as np

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


def train_hf(
    model_name: str,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
) -> Dict[str, float]:
    if data_args is None:
        data_args = GlueDataTrainingArguments(task_name="sts-b", data_dir="./data/combined")
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

    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            return glue_compute_metrics(
                task_name, np.squeeze(p.predictions), p.label_ids
            )

        return compute_metrics_fn

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
