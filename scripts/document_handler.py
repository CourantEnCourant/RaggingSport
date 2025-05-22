from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging


class DocumentHandler:
    def __init__(self):
        self.documents: list[Document] | None = None

    def load_documents(self, directory_path: str) -> "DocumentHandler":
        """Load text documents from a directory"""
        try:
            loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
            self.documents = loader.load()
            logging.info(f"Loaded {len(self.documents)} documents from {directory_path}")
            return self
        except Exception as e:
            raise Exception(e)

    def split_documents(self, chunk_size=1000, chunk_overlap=200) -> "DocumentHandler":
        """Split documents into manageable chunks"""
        if self.documents is None:
            raise Exception("Document not loaded. Use load_documents() first.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.documents = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.documents)} chunks")
        return self

    def get_documents(self) -> list[Document]:
        return self.documents
    