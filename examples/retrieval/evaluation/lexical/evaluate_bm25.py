"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR.
To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_bm25.py
"""
import argparse

from beir.configs import dataset_stored_loc
from beir.custom_logging import setup_logger
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import os
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="test")
    params = parser.parse_args()

    # Download scifact.zip dataset and unzip the dataset
    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    # Provide the data path where scifact has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) scifact/corpus.jsonl  (format: jsonlines)
    # (2) scifact/queries.jsonl (format: jsonlines)
    # (3) scifact/qrels/test.tsv (format: tsv ("\t"))
    corpus, queries, qrels = GenericDataLoader(data_path).load(split=params.split)

    #### Lexical Retrieval using Bm25 (Elasticsearch) ####
    #### Provide a hostname (localhost) to connect to ES instance
    #### Define a new index name or use an already existing one.
    #### We use default ES settings for retrieval
    #### https://www.elastic.co/

    hostname = "localhost"
    index_name = f"beir-{dataset}"

    #### Intialize ####
    # (1) True - Delete existing index and re-index all documents from scratch
    # (2) False - Load existing index
    initialize = True  # False

    #### Sharding ####
    # (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1
    # SciFact is a relatively small dataset! (limit shards to 1)
    number_of_shards = 1
    model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

    # (2) For datasets with big corpus ==> keep default configuration
    # model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
    retriever = EvaluateRetrieval(model)

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
