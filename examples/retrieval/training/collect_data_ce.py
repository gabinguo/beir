from beir.datasets.data_loader import GenericDataLoader
from beir.configs import dataset_stored_loc
from sentence_transformers import InputExample
from typing import List
from tqdm import tqdm
import requests
import json
import os
import pandas as pd
import argparse


def collect_training_data(dataset_name: str, limit: int, split: str = "train", number_positives: int = 1, number_negatives: int = 4):
    index_name = "beir/test"
    docker_beir_pyserini = "http://localhost:8000"
    data_folder: str = os.path.join(dataset_stored_loc, dataset_name)
    corpus, queries, qrels = GenericDataLoader(data_folder).load(split=split)
    query_ids: List[str] = list(qrels)

    positive_examples: List[InputExample] = []
    negative_examples: List[InputExample] = []
    for query_id in tqdm(query_ids[:limit], desc="Collecting training data...", total=limit, leave=True):
        positive_document_ids: List[str] = list(qrels[query_id])
        query_text: str = queries[query_id]
        # collect positive examples
        for positive_document_id in positive_document_ids[:number_positives]:
            positive_examples.append(InputExample(
                texts=[query_text, corpus[positive_document_id]["text"]], label=1
            ))
        # collect negative examples (pyserini)
        # use query/document to retrieve relative documents
        queries_for_search = [f"{corpus[pos_id]['title']} {query_text}" for pos_id in positive_document_ids[:number_positives]]
        payload = {
            "queries": queries_for_search,
            "qids": [f"{query_id}_{idx + 1}" for idx in range(len(queries_for_search))],
            "k": number_negatives + 1
        }
        hits = json.loads(
            requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

        count: int = 0
        for query_for_search in range(len(positive_document_ids)):
            if f"{query_id}_{query_for_search + 1}" in hits:
                for hit_id in hits[f"{query_id}_{query_for_search + 1}"]:
                    if hit_id not in positive_document_ids and count < number_negatives:
                        negative_document = corpus[hit_id]["title"] + " " + corpus[hit_id]["text"]
                        negative_examples.append(InputExample(
                            texts=[query_text, negative_document], label=0
                        ))
                        count += 1

    curr_folder = os.path.abspath(os.path.dirname(__file__))
    os.makedirs(os.path.join(curr_folder, dataset_name), exist_ok=True)
    with open(os.path.join(curr_folder, dataset_name, f"{dataset_name}_train_{limit}.jsonl"), 'w') as f:
        for example in [*positive_examples, *negative_examples]:
            f.write(json.dumps({
                "query": example.texts[0],
                "document": example.texts[1],
                "label": example.label
            }))
            f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--number_positives", type=int, default=1)
    parser.add_argument("--number_negatives", type=int, default=4)

    params = parser.parse_args()

    collect_training_data(
        dataset_name=params.dataset,
        limit=params.limit,
        split=params.split,
        number_positives=params.number_positives,
        number_negatives=params.number_negatives,
    )
