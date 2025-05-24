from langchain_ollama import OllamaLLM
import torch

from rag.vectorstore import prepare_vectorstore
from rag.qa_chain import QAChain
from qa import QA


class Answerer:
    def __init__(self, llm_name="tinyllama:1.1b",
                 vectorstore=prepare_vectorstore(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                                                 document_directory="../data/corpus",
                                                 device='cuda' if torch.cuda.is_available() else 'cpu')):
        self.llm = OllamaLLM(
            model=llm_name,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15,
            verbose=True
        )
        self.vectorstore = vectorstore
        self.qa_chain = QAChain(self.vectorstore, self.llm)

    def answer(self, qa: QA) -> QA:
        response = self.qa_chain.qa_with_sources(query=qa.question)
        qa.actual_output = response.answer
        return qa
