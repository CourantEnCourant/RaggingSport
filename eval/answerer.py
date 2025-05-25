from rag.qa_chain import QAChain
from qa import QA


class Answerer(QAChain):
    def answer(self, qa: QA) -> QA:
        response = self.qa_chain.qa_with_sources(query=qa.question)
        new_qa = QA(question=qa.question,
                    context=qa.context,
                    expected_output=qa.expected_output,
                    actual_output = response.answer,
                    retrieval_context=[document["page_content"] for document in response.retrieved_documents])
        return new_qa
