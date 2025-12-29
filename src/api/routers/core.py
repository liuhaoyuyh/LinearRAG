import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas import EvaluateRequest, IndexRequest, QARequest
from src.api.utils.path_utils import build_output_dir, resolve_path
from src.api.utils.rag_utils import build_rag_and_models, ensure_index_ready, load_passages
from src.evaluate import Evaluator
from src.utils import setup_logging, LLM_Model

router = APIRouter()


@router.get("/health")
def health_check():
    """健康检查接口。"""
    # TODO
    return {"status": "ok"}


@router.post("/index")
def run_index(request: IndexRequest):
    """构建索引并返回路径信息。"""
    try:
        passages = load_passages(request.dataset_name)
        output_dir = build_output_dir(request.dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))
        rag, _ = build_rag_and_models(request)

        rag.index(passages)
        dataset_dir = os.path.join(str(resolve_path(request.working_dir)), request.dataset_name)
        response = {
            "status": "success",
            "dataset": request.dataset_name,
            "working_dir": str(resolve_path(request.working_dir)),
            "graph_path": str(Path(dataset_dir) / "LinearRAG.graphml"),
            "ner_path": str(Path(dataset_dir) / "ner_results.json"),
            "embeddings": {
                "passage": str(Path(dataset_dir) / "passage_embedding.parquet"),
                "entity": str(Path(dataset_dir) / "entity_embedding.parquet"),
                "sentence": str(Path(dataset_dir) / "sentence_embedding.parquet"),
            },
            "log_path": os.path.join(output_dir, "log.txt"),
        }
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qa")
def run_qa(request: QARequest):
    """基于索引进行问答。"""
    try:
        output_dir = build_output_dir(request.dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))
        rag, _ = build_rag_and_models(request)
        ensure_index_ready(rag)
        qa_results = rag.qa([q.model_dump() for q in request.questions])
        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=4)
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "predictions_path": predictions_path,
            "results": qa_results,
            "log_path": os.path.join(output_dir, "log.txt"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
def run_evaluate(request: EvaluateRequest):
    """评估预测结果并输出指标。"""
    try:
        if not os.path.exists(request.predictions_path):
            raise HTTPException(status_code=400, detail=f"未找到预测结果文件：{request.predictions_path}")
        output_dir = os.path.dirname(request.predictions_path)
        setup_logging(os.path.join(output_dir, "log.txt"))
        llm_model = LLM_Model(request.llm_model)
        evaluator = Evaluator(llm_model=llm_model, predictions_path=request.predictions_path)
        llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=request.max_workers)
        evaluation_path = os.path.join(output_dir, "evaluation_results.json")
        return {
            "status": "success",
            "dataset": request.dataset_name,
            "predictions_path": request.predictions_path,
            "evaluation_path": evaluation_path,
            "metrics": {
                "llm_accuracy": llm_accuracy,
                "contain_accuracy": contain_accuracy,
            },
            "log_path": os.path.join(output_dir, "log.txt"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
