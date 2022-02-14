from beir.configs import dataset_stored_loc
from beir.custom_logging import setup_logger
from beir.retrieval import models
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
    parser.add_argument("--query_encoder", type=str, default="facebook-dpr-question_encoder-multiset-base")
    parser.add_argument("--context_encoder", type=str, default="facebook-dpr-ctx_encoder-multiset-base")
    parser.add_argument("--batch_size", type=int, default=32)
    params = parser.parse_args()

    dataset = params.dataset
    data_path = os.path.join(dataset_stored_loc, dataset)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=params.split)

    model = DRES(models.SentenceBERT((
        params.query_encoder,
        params.context_encoder,
        " [SEP] "), batch_size=params.batch_size))
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    results = retriever.retrieve(corpus, queries)

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logger.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    # #### Print top-k documents retrieved ####
    # top_k = 10
    #
    # query_id, ranking_scores = random.choice(list(results.items()))
    # scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    # logging.info("Query : %s\n" % queries[query_id])
    #
    # for rank in range(top_k):
    #     doc_id = scores_sorted[rank][0]
    #     # Format: Rank x: ID [Title] Body
    #     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
