from abstract_metric import AbstractMetric
from deepeval.test_case import LLMTestCase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingSimilarityMetric(AbstractMetric):
    def __init__(self, threshold=0.75, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def measure(self, test_case: LLMTestCase):
        context = " ".join(test_case.context)
        embeddings = self.model.encode([test_case.actual_output, context])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        test_case.metric_scores[self.name()] = float(score)
        return score

    def is_pass(self, score: float) -> bool:
        return score >= self.threshold

    def name(self):
        return "embedding_similarity"

    def rationale(self) -> str:
        return "Scores based on cosine similarity between context and generated answer"
