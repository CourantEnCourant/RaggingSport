from deepeval.metrics import (BaseMetric,
                              AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric)

from eval.custom_metrics.embedding_similarity_metric import EmbeddingSimilarityMetric


class MetricFactory:
    def create_metric(self, metric_type: str) -> BaseMetric:
        if metric_type == "answer_relevancy":
            return AnswerRelevancyMetric(
                threshold=0.7,
                model="gpt-4",
                include_reason=True
            )
        elif metric_type == "faithfulness":
            return FaithfulnessMetric(
                threshold=0.7,
                model="gpt-4",
                include_reason=True
            )
        elif metric_type == "contextual_precision":
            return ContextualPrecisionMetric(
                threshold=0.7,
                model="gpt-4",
                include_reason=True
            )
        elif metric_type == "contextual_recall":
            return ContextualRecallMetric(
                threshold=0.7,
                model="gpt-4",
                include_reason=True
            )
        elif metric_type == "contextual_relevancy":
            return ContextualRelevancyMetric(
                threshold=0.7,
                model="gpt-4",
                include_reason=True
            )
        elif metric_type == "embedding_similarity":
            return EmbeddingSimilarityMetric()
        else:
            raise self.MetricTypeNotSupportedException("Metric type not supported")

    class MetricTypeNotSupportedException(Exception):
        pass
