from deepeval.test_case import LLMTestCase
from qa import QA


class LLMTestCaseFactory:
    def create_test_case(self, qa: QA, metric_type: str):
        if metric_type == "answer_relevancy":
            return LLMTestCase(
                input=qa.question,
                actual_output=qa.actual_output
            )
        elif metric_type == "faithfulness":
            return LLMTestCase(
                input=qa.question,
                actual_output=qa.actual_output,
                retrieval_context=qa.retrieval_context
            )
        elif metric_type == "contextual_precision":
            return LLMTestCase(
                input=qa.question,
                actual_output=qa.actual_output,
                expected_output=qa.expected_output,
                retrieval_context=qa.retrieval_context
            )
        elif metric_type == "contextual_recall":
            return LLMTestCase(
                input=qa.question,
                actual_output=qa.actual_output,
                expected_output=qa.expected_output,
                retrieval_context=qa.retrieval_context
            )
        elif metric_type == "contextual_relevancy":
            return LLMTestCase(
                input=qa.question,
                actual_output=qa.actual_output,
                retrieval_context=qa.retrieval_context
            )
        elif metric_type == "embedding_similarity":
            return LLMTestCase(
                input=qa.question,
                context=qa.retrieval_context,
                expected_output=qa.expected_output,
                actual_output=qa.actual_output
            )
        elif metric_type == "tfidf_similarity":
            return LLMTestCase(
                input=qa.question,
                expected_output=qa.expected_output,
                actual_output=qa.actual_output
            )
        else:
            raise self.MetricTypeNotSupportedException("Metric type not supported")

    def create_test_cases(self, qas: list[QA], metric_type: str):
        return [self.create_test_case(qa, metric_type) for qa in qas]

    class MetricTypeNotSupportedException(Exception):
        pass
