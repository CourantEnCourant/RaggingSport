from qa_database_handler import QADatabaseHandler

from pathlib import Path


def main(contexts_dir: Path, output_dir: Path):
    # Prepare contexts
    contexts = [context_file.read_text() for context_file in contexts_dir.glob("*.txt")]
    # Create and save qas
    qa_database = QADatabaseHandler()
    qa_database.generate_qas(contexts)
    qa_database.save_qas(output_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contexts_dir', type=Path, default=Path("../data/corpus"))
    parser.add_argument('--output_dir', type=Path, default=Path("../data/qa"))
    args = parser.parse_args()

    main(contexts_dir=args.contexts_dir, output_dir=args.output_dir)
