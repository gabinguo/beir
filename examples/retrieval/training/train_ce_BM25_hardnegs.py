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
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from beir.configs import default_ranker, dataset_stored_loc
from beir.custom_logging import setup_logger, log_map
from beir.datasets.data_loader import GenericDataLoader
import os, tqdm
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ranker", type=str, default=default_ranker)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_hard_negs", type=int, default=4)
    parser.add_argument("--limit", type=int, default=100)
    params = parser.parse_args()

    log_map(logger, "Arguments", params.__dict__)

    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=params.split)

    queries = dict(itertools.islice(queries.items(), params.limit))
    qrels = dict(itertools.islice(qrels.items(), params.limit))

    docker_beir_pyserini = "http://localhost:8000"
    index_name = "beir/test"

    # start collecting training data
    triplets = []
    qids = list(qrels)
    hard_negatives_max = params.num_hard_negs

    for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25", leave=False):
        query_id, query_text = qids[idx], queries[qids[idx]]
        # get the ids of positive documents
        pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        # use title + document to retrieve
        pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
        # use query to retrieve
        # pos_doc_texts = [corpus[doc_id]["title"] + " " + query_text for doc_id in pos_docs]
        # Pyserini
        payload = {
            "queries": pos_doc_texts,
            "qids": [f"query_id_{query_idx + 1}" for query_idx in range(len(pos_doc_texts))],
            "k": hard_negatives_max + 1
        }
        hits = json.loads(
            requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

        for hit in hits:
            hits[hit] = dict(itertools.islice(hits[hit].items(), hard_negatives_max + 1))

        for pos_idx, pos_text in enumerate(pos_doc_texts):
            que_id = f"query_id_{pos_idx + 1}"
            for neg_id in hits[que_id]:
                if neg_id not in pos_docs:
                    neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                    triplets.append([query_text, pos_text, neg_text])

    logger.info(f"Gathered {len(triplets)} triplets for training.")

    logger.info("Prepare for training")
    #### Provide any sentence-transformers or HF model
    model_name = params.ranker
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    train_examples = []
    record_pos = set()
    record_negs = dict()
    for triplet in tqdm.tqdm(triplets, desc="Convert triplets to CE training data: ", leave=False):
        query_text, pos_text, neg_text = triplet
        if query_text not in record_pos:
            record_pos.add(query_text)
            train_examples.append(InputExample(
                texts=[query_text, pos_text], label=0.99
            ))
        if neg_text not in record_negs[query_text]:
            if record_negs[query_text]:
                record_negs[query_text].append(neg_text)
            else:
                record_negs[query_text] = [neg_text]
            train_examples.append(InputExample(
                texts=[query_text, neg_text], label=0.01
            ))

    logger.info(f"Convert {len(triplets)} triplets to {len(train_examples)} input examples.")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=params.batch_size)
    # Provide model save path
    model_save_path = f"{model_name}_{dataset}_bm25_{params.limit}_hard_negs"
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f"Model Saved to: {model_save_path}")

    #### Configure Train params
    num_epochs = 2
    evaluation_steps = 10_000
    warmup_steps = int(len(train_examples) * num_epochs / params.batch_size * 0.1)

    model.fit(train_dataloader=train_dataloader,
              epochs=num_epochs,
              output_path=model_save_path,
              warmup_steps=warmup_steps,
              evaluation_steps=evaluation_steps,
              use_amp=True)

    model.save_pretrained(model_save_path)
