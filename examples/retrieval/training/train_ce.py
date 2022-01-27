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
import json

from torch.utils.data import DataLoader
from sentence_transformers import losses, models, SentenceTransformer, CrossEncoder, InputExample
from beir.configs import dataset_stored_loc, basedir
from beir.custom_logging import setup_logger, log_map
import os
import tqdm
import logging

logger = logging.getLogger(__name__)
setup_logger(logger)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-6)
    params = parser.parse_args()

    dataset = params.dataset
    limit = params.limit
    split = params.split
    lr: float = params.lr
    batch_size: int = params.batch_size
    default_ranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    train_file: str = params.train_file

    log_map(logger, "Arguments", params.__dict__)

    model_name = default_ranker
    model = CrossEncoder(model_name, num_labels=1, max_length=256)
    train_examples = []
    with open(train_file) as f:
        for line in f:
            instance = json.loads(line)
            train_examples.append(InputExample(
                texts=[instance["query"], instance["document"]],
                label=instance["label"]
            ))





    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    # Provide model save path
    model_save_path = f"./ranker_{dataset}_bm25_{limit}"
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f"Model Saved to: {model_save_path}")

    #### Configure Train params
    num_epochs = 2
    evaluation_steps = 10_000
    # warmup_steps = int(len(train_examples) * num_epochs / batch_size * 0.1)
    warmup_steps = 0

    model.fit(train_dataloader=train_dataloader,
                  epochs=num_epochs,
                  output_path=model_save_path,
                  warmup_steps=warmup_steps,
                  optimizer_params={"lr": lr},
                  evaluation_steps=evaluation_steps,
                  use_amp=True)

    model.save_pretrained(model_save_path)
