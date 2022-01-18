import logging
import os
from beir import util
from beir.configs import dataset_stored_loc
from beir.custom_logging import setup_logger

logger = logging.getLogger(__name__)
setup_logger(logger)


def main():
    os.makedirs(dataset_stored_loc, exist_ok=True)
    # dataset_files = ["msmarco.zip", "trec-covid.zip", "nfcorpus.zip",
    #                  "nq.zip", "hotpotqa.zip", "fiqa.zip", "arguana.zip",
    #                  "webis-touche2020.zip", "cqadupstack.zip", "quora.zip",
    #                  "dbpedia-entity.zip", "scidocs.zip", "fever.zip",
    #                  "climate-fever.zip", "scifact.zip", "germanquad.zip"]

    # download the necessary datasets that one may need
    # dataset_files = ["nfcorpus.zip", "hotpotqa.zip", "fiqa.zip", "quora.zip", "fever.zip", "scifact.zip"]
    dataset_files = ["scifact"]

    for dataset in dataset_files:
        zip_file = os.path.join(dataset_stored_loc, f"{dataset}.zip")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)

        logger.info("Downloading {} ...".format(dataset))
        if not os.path.exists(zip_file):
            util.download_url(url, zip_file)

        logger.info("Unzipping {} ...".format(dataset))
        util.unzip(zip_file, dataset_stored_loc)


if __name__ == '__main__':
    main()
