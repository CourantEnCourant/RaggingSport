from dataclasses import dataclass, asdict


@dataclass
class Response:
    query: str
    answer: str
    retrieved_documents: list[dict[str, str]]

    def as_dict(self):
        return asdict(self)
