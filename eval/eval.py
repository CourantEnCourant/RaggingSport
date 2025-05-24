from deepeval import evaluate

from qa_database_handler import QADatabaseHandler
from llm_testcase_factory import LLMTestCaseFactory
from metric_factory import MetricFactory

from pathlib import Path
import json


def main(metric_type: str, output_path):
    # Load and get qas
    qa_database_handler = QADatabaseHandler()
    qas = qa_database_handler.load_qas().get_qas()
    # Prepare testcases and metric
    metric = MetricFactory().create_metric(metric_type)
    test_cases = LLMTestCaseFactory().create_test_cases(qas, metric_type)
    # Run evaluation
    results = evaluate(test_cases=test_cases, metrics=[metric])
    # Save results to JSON
    serializable_results = [result.__dict__ for result in results]
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=4)

    print(f"Saved results to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric_type', type=str, default='embedding_similarity')
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()

    main(args.metric_type, args.output_path)
