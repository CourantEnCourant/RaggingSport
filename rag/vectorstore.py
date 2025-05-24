from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag.document_handler import DocumentHandler

import logging
from tqdm import tqdm
from pathlib import Path


class VectorStore:
    def __init__(self):
        self.vectorstore = None

    def create_vectorstore(self, chunks: list[Document], embeddings: HuggingFaceEmbeddings,
                           batch_size: int = 2) -> None:
        faiss_index = None
        for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing documents"):
            batch = chunks[i:i + batch_size]
            if faiss_index is None:
                faiss_index = FAISS.from_documents(batch, embeddings)
            else:
                faiss_index.add_documents(batch)

        self.vectorstore = faiss_index

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


def prepare_vectorstore(embedding_model_name: str,
                        document_directory: str,
                        device: str,
                        vectorstore_directory=Path("../data/vectorstore")) -> VectorStore:
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = VectorStore()
    if (vectorstore_directory / "index.faiss").exists():
        vectorstore.load_vectorstore(str(vectorstore_directory), embeddings)
        print(f"Loaded vectorstore from {vectorstore_directory}")
    else:
        # Load documents
        print("Generating vectorstore")
        document_handler = DocumentHandler()
        document_handler.load_documents(document_directory).split_documents()
        # Create vectorstore and save
        vectorstore.create_vectorstore(chunks=document_handler.get_documents(),
                                       embeddings=embeddings)
        vectorstore.save_vectorstore(str(vectorstore_directory))
        print(f"Vectorstore created at {vectorstore_directory}")
    return vectorstore
