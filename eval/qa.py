from dataclasses import dataclass, field, asdict


@dataclass
class QA:
    """QA product"""
    question: str
    context: str
    expected_output: str
    actual_output: str = field(default="")
    retrieval_context: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
