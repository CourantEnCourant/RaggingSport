from transformers import pipeline
from deepeval.dataset import EvaluationDataset
from deepeval.evaluate import evaluate
from langchain_community.document_loaders import TextLoader
from deepeval.test_case import LLMTestCase

from eval.custom_metrics.embedding_similarity_metric import EmbeddingSimilarityMetric

loader = TextLoader("./data/") #METTEZ VOTRE DOCUMENT
data = loader.load()
documents = ## TRAITEZ VOS DONNEES


generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_synthetic_qa(context):
    q_prompt = f"Generate a question that can be answered by this text:\n{context}"
    a_prompt = f"Answer the following question based on the text:\n{context}"

    question = generator(q_prompt, max_new_tokens=50)[0]["generated_text"]
    answer = generator(f"{a_prompt}\nQuestion: {question}", max_new_tokens=80)[0]["generated_text"]

    return question.strip(), answer.strip()

samples = []
for doc in documents[5:6]:
    question, reference_answer = generate_synthetic_qa(doc)
    samples.append(LLMTestCase(
        input=question,
        context=[doc],
        expected_output=reference_answer,
        actual_output = #METTEZ ICI LA SORTIE DE VOTRE RAG
    ))

synthetic_dataset = EvaluationDataset(test_cases=samples)

metric = EmbeddingSimilarityMetric(threshold=0.75)
results = evaluate(dataset=synthetic_dataset, metrics=[metric])

print("Similarity Score:", results[0].metric_scores["embedding_similarity"])
