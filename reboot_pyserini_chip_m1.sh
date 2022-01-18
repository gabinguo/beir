docker stop pyserini
docker run --platform linux/amd64 --detach -p 8000:8000 --name pyserini -it --rm beir/pyserini-fastapi
