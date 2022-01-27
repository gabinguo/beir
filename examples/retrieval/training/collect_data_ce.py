from beir.datasets.data_loader import GenericDataLoader
from beir.configs import dataset_stored_loc
from beir.custom_logging import setup_logger, log_map
from sentence_transformers import InputExample
from typing import List, Set
from tqdm import tqdm
import requests
import json
import os
import argparse
import random
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)


def collect_training_data(number_positives: int = 1, number_random_negatives: int = 2, number_hard_negatives: int = 2):
    data_folder: str = os.path.join(dataset_stored_loc, dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder).load(split=split)
    query_ids: List[str] = list(qrels)
    corpus_ids: List[str] = list(corpus)

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
        queries_for_search = [f"{corpus[pos_id]['title']} {query_text}" for pos_id in
                              positive_document_ids[:number_positives]]
        payload = {
            "queries": queries_for_search,
            "qids": [f"{query_id}_{idx + 1}" for idx in range(len(queries_for_search))],
            "k": 100
        }
        hits = json.loads(
            requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

        cnt_hard_negatives: int = 0
        cnt_random_negatives: int = 0
        record_hit_ids: List[str] = []
        # Block to add hard negatives
        for query_for_search in range(len(positive_document_ids)):
            if f"{query_id}_{query_for_search + 1}" in hits:
                hit_ids = list(hits[f"{query_id}_{query_for_search + 1}"])[:100]
                record_hit_ids.extend(hit_ids)
                for hit_id in hit_ids:
                    if hit_id not in positive_document_ids and cnt_hard_negatives < number_hard_negatives:
                        negative_document = corpus[hit_id]["title"] + " " + corpus[hit_id]["text"]
                        negative_examples.append(InputExample(
                            texts=[query_text, negative_document], label=0
                        ))
                        cnt_hard_negatives += 1

        record_hit_ids: Set[str] = set(record_hit_ids)
        # Block to add random negatives
        while cnt_random_negatives < number_random_negatives:
            random_document_id = random.choice(corpus_ids)
            while random_document_id in record_hit_ids or random_document_id in positive_document_ids:
                logger.info("Overlapped with hard examples... random pick another one...")
                random_document_id = random.choice(corpus_ids)
            random_negative_document = corpus[random_document_id]["title"] + " " + corpus[random_document_id]["text"]
            negative_examples.append(InputExample(
                texts=[query_text, random_negative_document], label=0
            ))
            cnt_random_negatives += 1

        curr_folder = os.path.abspath(os.path.dirname(__file__))
        os.makedirs(os.path.join(curr_folder, dataset), exist_ok=True)
        with open(os.path.join(curr_folder, dataset, output_ds_name),
                  'w') as f:
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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--number_positives", type=int, default=1)
    parser.add_argument("--number_random_negatives", type=int, default=2)
    parser.add_argument("--number_hard_negatives", type=int, default=2)

    params = parser.parse_args()
    log_map(logger, "Arguments", params.__dict__)

    dataset = params.dataset
    split = params.split
    limit = params.limit
    num_pos = params.number_positives
    num_rands = params.number_random_negatives
    num_hards = params.number_hard_negatives

    output_ds_name: str = f"{dataset}_{split}_{limit}_rand_{num_rands}_hard_{num_hards}.jsonl"
    docker_beir_pyserini = f"http://localhost:{params.port}"
    collect_training_data(
        number_positives=num_pos,
        number_random_negatives=num_rands,
        number_hard_negatives=num_hards
    )
