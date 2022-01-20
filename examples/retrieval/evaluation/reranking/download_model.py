from beir.reranking.models import CrossEncoder
from beir.configs import default_ranker

cross_encoder_model = CrossEncoder(default_ranker)
cross_encoder_model.model.save("./default_ranker")
