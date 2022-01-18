docker stop pyserini
docker run --detach -p 8000:8000 --name pyserini -it --rm beir/pyserini-fastapi
