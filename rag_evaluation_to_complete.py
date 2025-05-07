from transformers import pipeline
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, ContextualPrecisionMetric
from deepeval.evaluate import evaluate
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import TextLoader
from deepeval.metrics.base_metric import BaseMetric
from deepeval.test_case import LLMTestCase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# loader = TextLoader("./data/") #METTEZ VOTRE DOCUMENT
# data = loader.load()
from datasets import load_dataset
ds = load_dataset("bilgeyucel/seven-wonders", split="train")
documents = ds['content'] ## TRAITEZ VOS DONNEES

generator = pipeline("text2text-generation", model="google/flan-t5-small", device=device)
#embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_synthetic_qa(context):
    q_prompt = f"Generate a question that can be answered by this text:\n{context}"
    a_prompt = f"Answer the following question based on the text:\n{context}"

    question = generator(q_prompt, max_new_tokens=50)[0]["generated_text"]
    answer = generator(f"{a_prompt}\nQuestion: {question}", max_new_tokens=80)[0]["generated_text"]

    return question.strip(), answer.strip()

import importlib
import rag
importlib.reload(rag)

samples = []
for doc in documents[5:6]:
    question, reference_answer = generate_synthetic_qa(doc)
    samples.append(LLMTestCase(
        input=question,
        context=[doc],
        expected_output=reference_answer,
        actual_output = rag.rag(question,"data") #METTEZ ICI LA SORTIE DE VOTRE RAG
    ))

synthetic_dataset = EvaluationDataset(test_cases=samples)

class EmbeddingSimilarityMetric(BaseMetric):
    def __init__(self, threshold=0.75, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    def measure(self, test_case: LLMTestCase):
        context = " ".join(test_case.context)
        embeddings = self.model.encode([test_case.actual_output, context])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        return score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)

    def is_successful(self):
        return self.success
    
    def is_pass(self, score: float) -> bool:
        return score >= self.threshold

    def name(self):
        return "embedding_similarity"

    def rationale(self) -> str:
        return "Scores based on cosine similarity between context and generated answer"

metric = EmbeddingSimilarityMetric(threshold=0.75)
results = synthetic_dataset.evaluate(metrics=[metric])

print(results)
# print("Similarity Score:", results[0].metric_scores["embedding_similarity"])