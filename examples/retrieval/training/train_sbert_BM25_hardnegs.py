"""
    These examples show how to train a Bi-Encoder for any BEIR dataset.

    The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
    These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

    For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
    (query, positive_passage, negative_passage)

    Negative passage are hard negative examples, that where retrieved by lexical search. We use Elasticsearch
    to get (max=10) hard negative examples given a positive passage.

    Running this script:
    python train_sbert_BM25_hardnegs.py
"""
import argparse
import itertools
import json

import requests
from sentence_transformers import losses, models, SentenceTransformer
from beir.configs import default_ranker, dataset_stored_loc
from beir.custom_logging import setup_logger
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, tqdm
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ranker", type=str, default=default_ranker)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=100)
    params = parser.parse_args()

    #### Download nfcorpus.zip dataset and unzip the dataset
    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=params.split)

    queries = dict(itertools.islice(queries.items(), params.limit))
    qrels = dict(itertools.islice(qrels.items(), params.limit))

    # Convert BEIR corpus to Pyserini Format #
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

    # make sure the docker beir pyserini container is on, (restarted)
    docker_beir_pyserini = "http://127.0.0.1:8000"

    # Upload Multipart-encoded files
    logger.info("[Start] Uploading to the Pyserini docker container...")
    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)
    logger.info("[Done_] Uploading to the Pyserini docker container...")

    logger.info("[Start] Indexing in the Pyserini docker container...")
    # Index documents to Pyserini #
    index_name = f"beir-{dataset}"
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})
    logger.info("[Done_] Indexing in the Pyserini docker container...")

    # start collecting training data
    triplets = []
    qids = list(qrels)
    hard_negatives_max = 10

    for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25", leave=False):
        query_id, query_text = qids[idx], queries[qids[idx]]
        pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        # use document to retrieve
        pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
        # use query to retrieve
        # TODO:
        # BM25
        # hits = bm25.retriever.es.lexical_multisearch(texts=pos_doc_texts, top_hits=hard_negatives_max + 1)

        # Pyserini
        payload = {"queries": pos_doc_texts, "qids": [query_id], "k": hard_negatives_max + 1}
        hits = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)[
            "results"]

        for pos_text in pos_doc_texts:
            for neg_id in hits[query_id]:
                if neg_id not in pos_docs:
                    neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                    triplets.append([query_text, pos_text, neg_text])

    #### Provide any sentence-transformers or HF model
    model_name = params.ranker
    word_embedding_model = models.Transformer(model_name, max_seq_length=300)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    #### Provide a high batch-size to train better with triplets!
    retriever = TrainRetriever(model=model, batch_size=params.batch_size)

    #### Prepare triplets samples
    train_samples = retriever.load_train_triplets(triplets=triplets)
    train_dataloader = retriever.prepare_train_triplets(train_samples)

    #### Training SBERT with cosine-product
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    #### If no dev set is present from above use dummy evaluator
    ir_evaluator = retriever.load_dummy_evaluator()

    #### Provide model save path
    model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output",
                                   "{}-v2-{}-bm25-{}-hard-negs".format(model_name, dataset, params.limit))
    os.makedirs(model_save_path, exist_ok=True)

    #### Configure Train params
    num_epochs = 1
    evaluation_steps = 10000
    warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

    retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=ir_evaluator,
                  epochs=num_epochs,
                  output_path=model_save_path,
                  warmup_steps=warmup_steps,
                  evaluation_steps=evaluation_steps,
                  use_amp=True)
