from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import logging


class VectorStore:
    def __init__(self):
        self.vectorstore = None

    def create_vectorstore(self, chunks: list[Document], embeddings: HuggingFaceEmbeddings) -> None:
        self.vectorstore = FAISS.from_documents(chunks, embeddings)

    def save_vectorstore(self, path: str) -> None:
        """Save the FAISS vectorstore to disk"""
        try:
            self.vectorstore.save_local(path)
            logging.info(f"Saved vectorstore to {path}")
        except Exception as e:
            print(f"Error saving vectorstore: {e}")

    def load_vectorstore(self, path: str, embeddings: HuggingFaceEmbeddings) -> None:
        """Load the FAISS vectorstore from disk"""
        try:
            # Load the index
            self.vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            logging.info(f"Loaded FAISS index from {path}")
        except Exception as e:
            raise Exception(e)

    def as_retriever(self, **search_kwargs):
        """Method implemented to unify interface"""
        return self.vectorstore.as_retriever(**search_kwargs)
