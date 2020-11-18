from typing import Callable, Dict
import numpy as np
from scipy.stats import pearsonr
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
def evaluate(x, y):
    return pearsonr(x, y)[0]

def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)

def train_hf(
    model_name: str,
    data_args: GlueDataTrainingArguments = None,
    config: AutoConfig = None,
    train_args: TrainingArguments = None,
    checkpoint_args: from_checkpoint=False, checkpoint_path="./hf_models/",
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

    if from_checkpoint==True:
        checkpoints=next(os.walk(checkpoint_path+model_name))[1]
        #tu albo ta wartosc sie zmieni na sciezke, a jak nie wejdzie w ifa to zostanie sama nazwa
        #https://discuss.huggingface.co/t/loading-model-from-checkpoint-after-error-in-training/758/3
        model_name="./"+model_name+"/"+str(max(checkpoints,key=extract_number))
        
    
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
