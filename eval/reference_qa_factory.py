from langchain_ollama import OllamaLLM

from qa import QA


class ReferenceQAFactory:
    """Generate question and expected_output from context using simple factory pattern"""
    def __init__(self, llm_name="tinyllama:1.1b"):
        self.llm = OllamaLLM(
            model=llm_name,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15,
            verbose=True
        )
        self.context: str | None = None
        self.question: str | None = None
        self.expected_output: str | None = None

    def set_context(self, context: str) -> "ReferenceQAFactory":
        self.context = context
        return self

    def create_question(self) -> "ReferenceQAFactory":
        if self.context is None:
            raise self.ContextMissingException("Context not provided yet. Use self.define_context() method to create context")
        print("Generating question...")
        q_prompt = (f"Generate a question that can be answered by this text: {self.context}\n\n"
                    f"Output only the question generated.\n\n"
                    f"Ask the question as someone who has no knowledge of the text.")

        self.question = self.llm.invoke(q_prompt)
        print("Question generated")
        return self

    def create_answer(self) -> "ReferenceQAFactory":
        if self.question is None:
            raise self.QuestionMissingException("Question not provided yet. Use self.create_question() to create question")
        print("Generating reference answer...")
        a_prompt = (f'Answer the following question: {self.question} based on following text: {self.context}.\n\n'
                    f'Be concise about your answer.')
        self.expected_output = self.llm.invoke(a_prompt)
        print("Reference answer generated")
        return self

    def generate_qa_reference(self, context: str) -> QA:
        self.set_context(context).create_question().create_answer()
        return QA(question=self.question, context=self.context, expected_output=self.expected_output)

    class ContextMissingException(Exception):
        pass

    class QuestionMissingException(Exception):
        pass
