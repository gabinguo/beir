from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore
import argparse
from beir.custom_logging import log_map, setup_logger
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Path to the train/dev files")
    parser.add_argument("--train_file", type=str, help="train filename")
    parser.add_argument("--dev_file", type=str, help="dev filename", default=None)
    parser.add_argument("--test_file", type=str, help="test filename", default=None)
    parser.add_argument("--max_query_len", type=int, default=64)
    parser.add_argument("--max_passage_len", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--number_positives", type=int, default=1)
    parser.add_argument("--number_negatives", type=int, default=1)
    parser.add_argument("--save_dir", type=str, help="path to output the fine-tuned model")
    query_model = "sentence-transformers/facebook-dpr-question_encoder-multiset-base"
    passage_model = "sentence-transformers/facebook-dpr-ctx_encoder-multiset-base"
    params = parser.parse_args()
    log_map(logger, "Arguments", params._dict__)

    retriever = DensePassageRetriever(
        document_store=InMemoryDocumentStore(),
        query_embedding_model=query_model,
        passage_embedding_model=passage_model,
    )

    retriever.train(
        data_dir=params.data_folder,
        train_filename=params.train_file,
        dev_filename=params.dev_file,
        test_filename=params.test_file,
        n_epochs=params.num_epochs,
        batch_size=params.batch_size,
        grad_acc_steps=params.grad_acc_steps,
        evaluate_every=100_000_000,  # no evaluation during training
        embed_title=False,
        num_positives=params.number_positives,
        num_hard_negatives=params.number_negatives,
        num_warmup_steps=0,
        save_dir=params.save_dir
    )
