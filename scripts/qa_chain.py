from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

    def qa_with_sources(self, query_str) -> dict:

        docs = self.retriever.invoke(query_str)
        # Convert Document to json-serializable dict
        docs_dict = [{"page_content": doc.page_content,
                      "metadata": doc.metadata}
                     for doc in docs]
        logging.info(f"Retrieved {len(docs)} documents")
        answer = self.qa_chain.invoke(query_str)
        return {"query": query_str, "result": answer, "source_documents": docs_dict}

    def format_docs(self, docs):
        return "\n\n".join([doc.page_content for doc in docs])
