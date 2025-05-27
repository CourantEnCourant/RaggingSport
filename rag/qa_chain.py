from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.response import Response

import logging


template = """Answer the question based on the following context:

Context: {context}

Question: {question}

Answer: """


class QAChain:
    def __init__(self, vectorstore, llm, template=template):
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(template)
            | llm
            | StrOutputParser()
        )

    def qa_with_sources(self, query) -> Response:

        docs = self.retriever.invoke(query)
        # Convert Document to json-serializable dict
        docs_dict = [{"page_content": doc.page_content,
                      "metadata": doc.metadata}
                     for doc in docs]
        logging.info(f"Retrieved {len(docs)} documents")
        answer = self.qa_chain.invoke(query)
        return Response(query, answer, docs_dict)

    @staticmethod
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
