import json
import os
import sys
import tempfile
from types import SimpleNamespace


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    os.environ["LLM_PROVIDER"] = "mock"

    from src.utils import LLM_Model
    from src.evaluate import Evaluator
    from src.LinearRAG import LinearRAG

    llm = LLM_Model("mock-model")

    rag = LinearRAG.__new__(LinearRAG)
    rag.config = SimpleNamespace(max_workers=2)
    rag.llm_model = llm

    def _mock_retrieve(questions):
        results = []
        for q in questions:
            question = q["question"] if isinstance(q, dict) else str(q)
            results.append(
                {
                    "question": question,
                    "sorted_passage": ["0:mock passage"],
                }
            )
        return results

    rag.retrieve = _mock_retrieve
    qa_results = LinearRAG.qa(rag, [{"question": "What is the capital of France?"}])
    assert qa_results[0]["pred_answer"] == "mock-answer", qa_results[0]["pred_answer"]

    with tempfile.TemporaryDirectory() as tmpdir:
        predictions_path = os.path.join(tmpdir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump([{"pred_answer": "Paris", "gold_answer": "Paris"}], f, ensure_ascii=False, indent=2)
        evaluator = Evaluator(llm_model=llm, predictions_path=predictions_path)
        llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=1)
        assert llm_accuracy == 1.0, llm_accuracy
        assert contain_accuracy == 1.0, contain_accuracy

    print("OK: mock provider validated (qa + evaluator).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
