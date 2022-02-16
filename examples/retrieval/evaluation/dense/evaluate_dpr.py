from beir.configs import dataset_stored_loc
from beir.custom_logging import setup_logger, log_map
from beir.retrieval.models import SentenceBERT
from beir.retrieval.models import DPR
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import os
import argparse

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--query_encoder", type=str, default="facebook/dpr-question_encoder-multiset-base")
    parser.add_argument("--context_encoder", type=str, default="facebook/dpr-ctx_encoder-multiset-base")
    parser.add_argument("--batch_size", type=int, default=128)
    params = parser.parse_args()

    log_map(logger, 'Params', params.__dict__)

    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=params.split)
    model = DRES(
        DPR((params.query_encoder, params.context_encoder)),
        batch_size=params.batch_size,
        corpus_chunk_size=params.batch_size * 4
    )
    retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1, 3, 5, 10])

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    logger.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
