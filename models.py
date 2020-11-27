all_models = [
    # "SparkBeyond/roberta-large-sts-b",
    "allenai/scibert_scivocab_uncased",
    "distilbert-base-uncased",
    "distilroberta-base",
    "bert-base-uncased",
    # "bert-large-uncased",
    # "t5-small",
    # "t5-base",
    # "t5-large",
    # "t5-3b",
    "albert-base-v2",
    # "albert-large-v2",
    "roberta-base",
    # "roberta-large",
    # "xlnet-base-cased",
    # "xlnet-large-cased",
]

siamese_models = [
    # "roberta-base-nli-stsb-mean-tokens",
    "bert-large-nli-stsb-mean-tokens",
    # "distilbert-base-nli-stsb-mean-tokens"
]

t5_models = [
    "t5-small",
    "t5-base",
    # "t5-large",
    # "t5-3b",  
]

directory_to_model_name = {"allenai_scibert_scivocab_uncased": "allenai/scibert_scivocab_uncased",
    "SparkBeyond_roberta-large-sts-b": "SparkBeyond/roberta-large-sts-b",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilroberta-base": "distilroberta-base",
    "bert-base-uncased": "bert-base-uncased",
    "bert-large-uncased": "bert-large-uncased",
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    "t5-3b": "t5-3b",
    "albert-base-v2": "albert-base-v2",
    "albert-large-v2": "albert-large-v2",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
    "xlnet-base-cased": "xlnet-base-cased",
    "xlnet-large-cased": "xlnet-large-cased",}