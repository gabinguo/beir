"""
This example shows how to evaluate DocTTTTTquery in BEIR.

Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, you can start the needed docker container with the following command:
docker run -p 8000:8000 -it --rm beir/pyserini-fastapi

Once the docker container is up and running in local, now run the code below.

For the example, we use the "castorini/doc2query-t5-base-msmarco" model for query generation.
In this example, We generate 3 questions per passage and append them with passage used for BM25 retrieval.  

Usage: python evaluate_anserini_docT5query.py
"""
import argparse

from beir import util
from beir.custom_logging import setup_logger
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.generation.models import QGenModel
from beir.configs import default_generator, dataset_stored_loc
from tqdm.autonotebook import trange

import pathlib, os, json
import logging
import requests

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--generator", type=str, default=default_generator)
    params = parser.parse_args()

    #### Download scifact.zip dataset and unzip the dataset
    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=params.split)
    corpus_ids = list(corpus.keys())
    corpus_list = [corpus[doc_id] for doc_id in corpus_ids]

    ################################
    #### 1. Question-Generation ####
    ################################

    #### docTTTTTquery model to generate synthetic questions.
    #### Synthetic questions will get prepended with document.
    #### Ref: https://github.com/castorini/docTTTTTquery

    model_path = params.generator
    qgen_model = QGenModel(model_path, use_fast=False)

    gen_queries = {}
    num_return_sequences = 40
    batch_size = 8

    for start_idx in trange(0, len(corpus_list), batch_size, desc='question-generation'):

        size = len(corpus_list[start_idx:start_idx + batch_size])
        ques = qgen_model.generate(
            corpus=corpus_list[start_idx:start_idx + batch_size],
            ques_per_passage=num_return_sequences,
            max_length=64,
            top_p=0.95,
            top_k=10)

        assert len(ques) == size * num_return_sequences

        for idx in range(size):
            start_id = idx * num_return_sequences
            end_id = start_id + num_return_sequences
            gen_queries[corpus_ids[start_idx + idx]] = ques[start_id: end_id]

    #### Convert BEIR corpus to Pyserini Format #####
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            query_text = " ".join(gen_queries[doc_id])
            data = {"id": doc_id, "title": title, "contents": text, "queries": query_text}
            json.dump(data, fOut)
            fOut.write('\n')

    #### Download Docker Image beir/pyserini-fastapi ####
    #### Locally run the docker Image + FastAPI ####
    docker_beir_pyserini = "http://127.0.0.1:8000"

    #### Upload Multipart-encoded files ####
    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

    #### Index documents to Pyserini #####
    index_name = f"beir-{dataset}"  # beir/scifact
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

    ######################################
    #### 2. Pyserini-Retrieval (BM25) ####
    ######################################

    #### Retrieve documents from Pyserini #####
    retriever = EvaluateRetrieval()
    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values),
               "fields": {"contents": 1.0, "title": 1.0, "queries": 1.0}}

    #### Retrieve pyserini results (format of results is identical to qrels)
    results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

    #### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
    # results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logger.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
