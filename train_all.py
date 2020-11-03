from sts_dev import train_hf
import json
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import STSBenchmarkDataReader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datetime import datetime
import csv
import os

all_models = [
    "SparkBeyond/roberta-large-sts-b",
    "bert-base-uncased",
    "gpt2",
    "albert-base-v2",
    "xlnet-large-cased",
]

sts_reader = STSBenchmarkDataReader("data/combined", normalize_scores=True, s1_col_idx=-3, s2_col_idx=-2, score_col_idx=-1)

for dataset_type in ["train", "test"]:
    with open(f"./data/combined/{dataset_type}.tsv") as f:
        fcsv_h = list(csv.reader(f, delimiter="\t"))
    with open(f"./data/combined/{dataset_type}_st.tsv", "w") as f:
        fcsv = csv.writer(f, delimiter="\t")
        fcsv.writerows(fcsv_h[1:])

for model_name in all_models:
    result = train_hf(model_name)
    with open(f"./hf_results/result_{model_name.replace('/', '_')}.json", "w") as f:
        json.dump(result, f)

    model = SentenceTransformer(model_name)
    model_save_path = f"./st_models/{model_name.replace('/', '_')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # Define your train dataset, the dataloader and the train loss
    train_data = SentencesDataset(
        sts_reader.get_examples("train_st.tsv"), model
    )
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=4,
        warmup_steps=100,
        output_path=model_save_path,
    )

    model = SentenceTransformer(model_save_path)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        sts_reader.get_examples("test_st.tsv"), name="sts-test"
    )
    test_evaluator(model, output_path=model_save_path)

for dataset_type in ["train", "test"]:
    os.remove(f"./data/combined/{dataset_type}_st.tsv")
