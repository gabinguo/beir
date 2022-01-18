import os

basedir: str = os.path.abspath(os.path.dirname(__file__))

# datasets store
dataset_stored_loc: str = os.path.join(basedir, "datasets_store")


# ranker
default_ranker: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# question generator
default_generator: str = "BeIR/query-gen-msmarco-t5-base-v1"