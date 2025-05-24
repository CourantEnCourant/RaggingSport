from reference_qa_factory import ReferenceQAFactory
from answerer import Answerer
from qa import QA

import json
from pathlib import Path


class QADatabaseHandler:
    def __init__(self, reference_qa_factory=ReferenceQAFactory(), answerer=Answerer()):
        self.reference_qa_factory = reference_qa_factory
        self.answerer = answerer
        self.qas: list[QA] | None = None

    def generate_qas(self, contexts: list[str]) -> "QADatabaseHandler":
        qa_references = [self.reference_qa_factory.generate_qa_reference(context) for context in contexts]
        qas = [self.answerer.answer(qa_reference) for qa_reference in qa_references]
        self.qas = qas
        return self

    def save_qas(self, output_dir=Path("../data/eval")) -> None:
        if self.qas is None:
            raise self.QANotCreatedError("QAs not created yet.")
        with open(output_dir / "qas.json", "w") as f:
            qas = [qa.to_dict() for qa in self.qas]
            json.dump(qas, f)

    def load_qas(self, input_path=Path("../data/eval/qas.json")) -> "QADatabaseHandler":
        if not input_path.exists():
            raise FileNotFoundError(f"No such file: {input_path}")
        with open(input_path, "r") as f:
            qas = json.load(f)
        self.qas = [QA(**qa) for qa in qas]
        return self

    def get_qas(self):
        if self.qas is None:
            raise self.QANotCreatedError("QAs not created yet.")
        return self.qas

    class QANotCreatedError(Exception):
        pass
