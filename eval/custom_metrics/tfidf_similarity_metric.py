from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSimilarityMetric(BaseMetric):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer()

    def measure(self, test_case: LLMTestCase):
        tfidf_matrix = self.vectorizer.fit_transform([test_case.actual_output, test_case.expected_output])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    def is_pass(self, score: float) -> bool:
        return score >= self.threshold

    def name(self):
        return "tfidf_similarity"

    def rationale(self) -> str:
        return "Scores based on tfidf vectorization and cosine similarity between actual output and expected output"
