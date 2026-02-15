"""Run the full evaluation: hybrid vs vector-only comparison."""

from src.config import Config
from src.eval.evaluator import Evaluator
from src.eval.queries import EVAL_QUERIES


def main():
    config = Config()
    evaluator = Evaluator(config)

    print("=" * 60)
    print("Running HYBRID evaluation (BM25 + Vector)...")
    print("=" * 60)
    hybrid_results = evaluator.run_full_eval(
        EVAL_QUERIES, label="hybrid"
    )

    print()
    print("=" * 60)
    print("Running VECTOR-ONLY evaluation (baseline)...")
    print("=" * 60)
    vector_results = evaluator.run_full_eval(
        EVAL_QUERIES,
        vector_weight=1.0,
        bm25_weight=0.0,
        label="vector_only",
    )

    print()
    print("=" * 60)
    print("Generating evaluation report...")
    print("=" * 60)
    report_path = config.project_root / "reports" / "evaluation_report.md"
    Evaluator.generate_report(hybrid_results, vector_results, report_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
