from sentence_transformers import SentenceTransformer, util
from models import siamese_models
import csv
import os

DATASET_SIZE = 50000
OFFSET = DATASET_SIZE * 7
TEST_SET = "./data/medical_test/vacc_sentences.tsv"
RESULT_DIRECTORY = "./with_scores/"
sentences1 = []
sentences2 = []
# with open(TEST_SET) as f:
#     fcsv = csv.reader(f, delimiter="\t")
#     next(fcsv)
#     for idx, l in enumerate(fcsv):
#         if idx < OFFSET:
#             continue
#         if idx == DATASET_SIZE + OFFSET:
#             break
#         sentences1.append(l[1])
#         sentences2.append(l[2])
sentences1 = [
"The reality is that vaccinations play a vital role in keeping kids healthy."]
sentences2 = [
"In the United States vaccines are suspected of  exposing kids to various diseases and risks."]
for model_name in siamese_models:
    model = SentenceTransformer(model_name)
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    with open(os.path.join(RESULT_DIRECTORY, f"single_siamese_{model_name}.tsv"), "a") as f:
        fcsv = csv.writer(f, delimiter="\t")
        for i in range(len(sentences1)):
            fcsv.writerow([sentences1[i], sentences2[i], float(cosine_scores[i][i])])
