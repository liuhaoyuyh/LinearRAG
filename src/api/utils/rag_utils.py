import json
import os
from pathlib import Path
from typing import List, Tuple

from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.utils import LLM_Model
from src.api.constants import BASE_DIR
from src.api.utils.path_utils import resolve_path


def load_passages(dataset_name: str) -> List[str]:
    """读取数据集 chunk 并组装检索 passages。"""
    dataset_root = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset"))
    dataset_dir = (dataset_root / dataset_name).resolve()
    chunks_path = dataset_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"未找到数据集文件：{chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return passages


def build_rag_and_models(request) -> Tuple[LinearRAG, LLM_Model]:
    """根据请求参数初始化 RAG 与 LLM。"""
    embedding_model_path = resolve_path(request.embedding_model, allow_missing=True)
    embedding_model = SentenceTransformer(str(embedding_model_path), device="cpu")
    llm_model = LLM_Model(request.llm_model)
    config = LinearRAGConfig(
        dataset_name=request.dataset_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model=request.spacy_model,
        working_dir=str(resolve_path(request.working_dir)),
        batch_size=request.batch_size,
        max_workers=request.max_workers,
        retrieval_top_k=request.retrieval_top_k,
        max_iterations=request.max_iterations,
        top_k_sentence=request.top_k_sentence,
        passage_ratio=request.passage_ratio,
        passage_node_weight=request.passage_node_weight,
        damping=request.damping,
        iteration_threshold=request.iteration_threshold,
    )
    return LinearRAG(global_config=config), llm_model


def ensure_index_ready(rag: LinearRAG) -> None:
    """确认索引已存在，否则抛出错误。"""
    if not rag.passage_embedding_store.hash_ids or not rag.entity_embedding_store.hash_ids:
        raise HTTPException(status_code=400, detail="未找到索引数据，请先调用 /index 完成索引构建。")


def chunk_texts(texts: List[str], chunk_size: int, overlap: int) -> List[str]:
    """将文本列表按词切分为重叠 chunk。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0:
        raise ValueError("overlap 不能为负数")
    tokens: List[str] = []
    for t in texts:
        if not t:
            continue
        tokens.extend(t.split())
    chunks = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        window = tokens[start:start + chunk_size]
        if not window:
            continue
        chunks.append(" ".join(window))
        if len(window) < chunk_size:
            break
    return chunks


def prefix_chunks(chunks: List[str]) -> List[str]:
    """为 chunk 列表添加索引前缀。"""
    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
