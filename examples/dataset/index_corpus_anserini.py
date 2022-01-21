from beir.custom_logging import setup_logger, log_map
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank
from beir.configs import default_ranker, dataset_stored_loc

import json
import requests
import os
import logging
import argparse

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--port", type=int)
    params = parser.parse_args()

    if not params.port:
        raise ValueError("Port not defined.")

    # Download trec-covid.zip dataset and unzip the dataset
    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    #### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) trec-covid/corpus.jsonl  (format: jsonlines)
    # (2) trec-covid/queries.jsonl (format: jsonlines)
    # (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=params.split)

    #########################################
    #### (1) RETRIEVE Top-100 docs using BM25 Pyserini,
    # Modified, originally it was pure BM25
    #########################################

    # Convert BEIR corpus to Pyserini Format #
    pyserini_jsonl = "pyserini.jsonl"
    with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
        for doc_id in corpus:
            title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
            data = {"id": doc_id, "title": title, "contents": text}
            json.dump(data, fOut)
            fOut.write('\n')

    # make sure the docker beir pyserini container is on, (restarted)
    docker_beir_pyserini = f"http://localhost:{params.port}"

    # Upload Multipart-encoded files
    logger.info("[Start] Uploading to the Pyserini docker container...")
    with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
        r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)
    logger.info("[Done_] Uploading to the Pyserini docker container...")

    logger.info("[Start] Indexing in the Pyserini docker container...")
    # Index documents to Pyserini #
    index_name = f"beir/test"
    r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})
    logger.info("[Done_] Indexing in the Pyserini docker container...")

