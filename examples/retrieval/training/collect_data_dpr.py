"""
    Collect data from pyserini container
"""
import json
import os
import random
import logging

import requests
from tqdm import tqdm

from beir.configs import dataset_stored_loc, basedir
from beir.datasets.data_loader import GenericDataLoader
from typing import List
import argparse
from beir.custom_logging import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


class CTXExample:
    def __init__(self, title: str, text: str, score: float, title_score: float, passage_id: str):
        self.title = title
        self.title_score = title_score
        self.text = text
        self.score = score
        self.passage_id = passage_id

    def decode(self):
        return {
            "title": self.title,
            "text": self.text,
            "score": self.score,
            "title_score": self.title_score,
            "passage_id": self.passage_id
        }


class DPRExample:

    def __init__(self, question: str, positive_ctxs: List[CTXExample],
                 negative_ctxs: List[CTXExample], dataset_name: str):
        self.question = question
        self.positive_ctxs = positive_ctxs
        self.negative_ctxs = negative_ctxs
        self.dataset = dataset_name

    def decode(self):
        return {
            "dataset": self.dataset,
            "question": self.question,
            "answers": [],
            "positive_ctxs": [ctx.decode() for ctx in self.positive_ctxs],
            "negative_ctxs": [],
            "hard_negative_ctxs": [ctx.decode() for ctx in self.negative_ctxs]
        }


def collect_training_data_for_DPR() -> List[dict]:
    data_folder: str = os.path.join(dataset_stored_loc, dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder).load(split=split)
    query_ids: List[str] = list(qrels)
    corpus_ids: List[str] = list(corpus)

    examples: List[DPRExample] = []

    for query_id in tqdm(query_ids[:limit], desc="Collecting DPR training data...", total=limit):
        positive_document_ids = list(qrels[query_id])
        question = queries[query_id]
        # collect examples
        pos_ctxs: List[CTXExample] = []
        neg_ctxs: List[CTXExample] = []
        for pos_doc_id in positive_document_ids[:number_positives]:
            title, text = corpus[pos_doc_id].values()
            pos_ctxs.append(CTXExample(title=title, text=text, score=1000, title_score=1, passage_id=pos_doc_id))
            # search for the hard negatives
            payload = {
                "queries": [f"{title} {question}"],
                "qids": [query_id],
                "k": 100
            }
            hits = json.loads(
                requests.post(docker_beir_pyserini + '/lexical/batch_search', json=payload).text)["results"]
            hit_ids = list(hits[query_id])[:100]  # take the top-100 as hard negatives

            cnt_hard: int = 0
            cnt_rand: int = 0
            # hard negatives collection
            for hit_id in hit_ids:
                if hit_id not in positive_document_ids and \
                        cnt_hard < number_hard_negatives:
                    title, text = corpus[hit_id].values()
                    neg_ctxs.append(CTXExample(title=title, text=text, score=0, title_score=0, passage_id=hit_id))
                    cnt_hard += 1

            # rand negatives collection
            while cnt_rand < number_rand_negatives:
                rand_doc_id = random.choice(corpus_ids)
                while rand_doc_id in hit_ids or rand_doc_id in positive_document_ids:
                    logger.info("Overlapped with hard/positive example..")
                    rand_doc_id = random.choice(corpus_ids)
                title, text = corpus[rand_doc_id].values()
                neg_ctxs.append(CTXExample(title=title, text=text, score=0, title_score=0, passage_id=rand_doc_id))
                cnt_rand += 1

        examples.append(DPRExample(
            question=question,
            positive_ctxs=pos_ctxs,
            negative_ctxs=neg_ctxs,
            dataset_name=dataset
        ))

    return [example.decode() for example in examples]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--number_positives", type=int, default=1)
    parser.add_argument("--number_hard_negatives", type=int, default=1)
    parser.add_argument("--number_rand_negatives", type=int, default=1)
    params = parser.parse_args()

    docker_beir_pyserini: str = f"http://localhost:{params.port}"
    dataset = params.dataset
    split = params.split
    limit = params.limit
    number_positives = params.number_positives
    number_hard_negatives = params.number_hard_negatives
    number_rand_negatives = params.number_rand_negatives

    examples = collect_training_data_for_DPR()
    os.makedirs(os.path.join(basedir, 'dpr_store'), exist_ok=True)
    with open(os.path.join(basedir, 'dpr_store', f'dpr_{dataset}_{split}_{limit}.json'), 'w') as f:
        json.dump(examples, f)
