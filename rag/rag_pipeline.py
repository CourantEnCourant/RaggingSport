from langchain_ollama import OllamaLLM
import torch

from vectorstore import prepare_vectorstore
from qa_chain import QAChain

import json
from pathlib import Path
import logging


def main(query: str,
         document_directory: str,
         embedding_model_name: str,
         llm_name: str,
         device: str,
         output_directory: Path):

    # Prepare vectorstore
    vectorstore = prepare_vectorstore(embedding_model_name, document_directory, device)

    # Prepare LLM
    llm = OllamaLLM(
            model=llm_name,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.15,
            verbose=True
        )

    # Prepare QAChain
    qa_chain = QAChain(vectorstore, llm)
    response = qa_chain.qa_with_sources(query)

    # Output to json
    with open(output_directory / 'qa_chain.json', 'w') as f:
        json.dump(response.as_dict(), f, indent=4)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('query', type=str)
    parser.add_argument('-d', '--document_directory', type=str, default="../data/corpus")
    parser.add_argument('-e', '--embedding_model_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('-l', '--llm_name', type=str, default="tinyllama:1.1b")
    parser.add_argument('-o', '--output_directory', type=Path, default=Path("../output"))

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Running on {device}")

    main(query=args.query,
         document_directory=args.document_directory,
         embedding_model_name=args.embedding_model_name,
         llm_name=args.llm_name,
         device=device,
         output_directory=args.output_directory)
