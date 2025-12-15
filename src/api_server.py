import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import re
import base64
import mimetypes
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import httpx

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.utils import LLM_Model, setup_logging
from prompts.loader import PromptLoader

BASE_DIR = Path(__file__).resolve().parent.parent


class BaseRunConfig(BaseModel):
    dataset_name: str = Field(..., description="数据集名称，需存在于 dataset/<name>/")
    embedding_model: str = Field("model/all-mpnet-base-v2", description="SentenceTransformer 模型名称或路径")
    spacy_model: str = Field("en_core_web_trf", description="spaCy 模型名称")
    llm_model: str = Field("gpt-4o-mini", description="OpenAI ChatCompletions 模型名称")
    working_dir: str = Field("./import", description="索引输出目录")
    batch_size: int = Field(128, ge=1)
    max_workers: int = Field(16, ge=1)
    retrieval_top_k: int = Field(5, ge=1)
    max_iterations: int = Field(3, ge=1)
    top_k_sentence: int = Field(1, ge=1)
    passage_ratio: float = Field(1.5, gt=0)
    passage_node_weight: float = Field(0.05, gt=0)
    damping: float = Field(0.5, gt=0, lt=1)
    iteration_threshold: float = Field(0.5, gt=0)


class IndexRequest(BaseRunConfig):
    """索引请求参数"""


class QuestionItem(BaseModel):
    question: str
    answer: Optional[str] = Field(default=None, description="可选：标注答案，用于后续评测")


class QARequest(BaseRunConfig):
    questions: List[QuestionItem]


class EvaluateRequest(BaseModel):
    dataset_name: str = Field(..., description="数据集名称，用于返回信息")
    predictions_path: str = Field(..., description="预测结果 JSON 路径")
    llm_model: str = Field("gpt-4o-mini", description="OpenAI ChatCompletions 模型名称")
    max_workers: int = Field(16, ge=1)


class MineruParseRequest(BaseModel):
    file_path: str = Field(..., description="本地文件路径（PDF/MD/图片等）")
    output_dir: Optional[str] = Field(None, description="解析结果输出目录，默认 results/mineru/<timestamp>/")
    # lang_list: Optional[List[str]] = Field(None, description="OCR 语言列表")
    backend: str = Field("pipeline", description="MinerU 后端类型")
    parse_method: str = Field("pipeline", description="PDF 解析方法")
    formula_enable: bool = Field(True, description="是否启用公式解析")
    table_enable: bool = Field(True, description="是否启用表格解析")
    server_url: Optional[str] = Field(None, description="VLM server URL，仅部分后端使用")
    return_md: bool = Field(True, description="是否返回 Markdown 内容")
    return_middle_json: bool = Field(False, description="是否返回中间 JSON")
    return_model_output: bool = Field(False, description="是否返回模型输出 JSON")
    return_content_list: bool = Field(False, description="是否返回内容列表 JSON")
    return_images: bool = Field(False, description="是否返回提取的图片")
    response_format_zip: bool = Field(False, description="是否以 ZIP 格式返回结果")
    start_page_id: int = Field(0, ge=0, description="PDF 解析起始页（从 0 开始）")
    end_page_id: int = Field(99999, ge=0, description="PDF 解析结束页（从 0 开始）")

class MindmapRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")

class ContentChunkRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")

class MarkdownChunkRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")

class MindmapExplainRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")
    dataset_name: Optional[str] = Field(
        default=None,
        description="可选：用于检索/索引的数据集名；默认等于 doc_name（对应 dataset/<name>/）",
    )
    embedding_model: str = Field("model/all-mpnet-base-v2", description="SentenceTransformer 模型名称或路径")
    spacy_model: str = Field("en_core_web_trf", description="spaCy 模型名称")
    llm_model: str = Field("gpt-4o-mini", description="OpenAI ChatCompletions 模型名称")
    working_dir: str = Field("./import", description="索引输出目录")
    batch_size: int = Field(128, ge=1)
    retrieval_top_k: int = Field(5, ge=1, description="每个模块检索返回的 top-k 上下文数量")
    max_workers: int = Field(16, ge=1, description="检索/索引并发参数（沿用 LinearRAG 配置）")
    max_iterations: int = Field(3, ge=1)
    top_k_sentence: int = Field(1, ge=1)
    passage_ratio: float = Field(1.5, gt=0)
    passage_node_weight: float = Field(0.05, gt=0)
    damping: float = Field(0.5, gt=0, lt=1)
    iteration_threshold: float = Field(0.5, gt=0)

    module_max_workers: int = Field(8, ge=1, description="模块解释并发数（LLM 调用并发）")
    include_tree: bool = Field(True, description="是否返回带回答的树结构")
    include_context: bool = Field(True, description="是否在结果中包含检索到的上下文")
    include_breadcrumb_in_query: bool = Field(
        False,
        description="是否在检索 query 中拼接节点路径（更强语境，可能更慢/更长）",
    )
    context_max_chars: int = Field(8000, ge=1, description="每个模块上下文最大字符数（拼接后）")
    context_per_passage_chars: int = Field(1500, ge=1, description="每段 passage 截断的最大字符数")


class MindmapExplainResultItem(BaseModel):
    id: str
    title: str
    level: int
    path: str
    module_type: Optional[str] = None
    anchor_title: Optional[str] = None
    subtree_node_count: Optional[int] = None
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    context: Optional[List[str]] = None
    llm_answer: Optional[str] = None
    error: Optional[str] = None


class MindmapExplainResponse(BaseModel):
    status: str
    doc_name: str
    dataset_name: str
    markdown_path: str
    module_count: int
    results: List[MindmapExplainResultItem]
    root: Optional[dict] = None
    log_path: Optional[str] = None

app = FastAPI(title="LinearRAG FastAPI", version="0.1.0", description="LinearRAG 图检索与问答服务化接口")


def _resolve_path(path_str: str, base: Path = BASE_DIR, allow_missing: bool = False) -> Path:
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    candidate = (base / path_obj).resolve()
    if candidate.exists() or allow_missing:
        return candidate
    return path_obj


def _load_passages(dataset_name: str) -> List[str]:
    dataset_root = Path(os.getenv("DATASET_DIR", BASE_DIR / "dataset"))
    dataset_dir = (dataset_root / dataset_name).resolve()
    chunks_path = dataset_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"未找到数据集文件：{chunks_path}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return passages


def _build_output_dir(dataset_name: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_root = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
    output_dir = (results_root / dataset_name / now).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def _build_mineru_output_dir(request_output_dir: Optional[str], file_name: str) -> Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results")) if request_output_dir is None else _resolve_path(
        request_output_dir, allow_missing=True)
    output_dir = (base_dir / "mineru" / file_name / now).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_latest_markdown(doc_name: str) -> Path:
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    md_path = latest_dir / doc_name / f"{doc_name}.md"
    if not md_path.exists():
        # 回退：尝试在最新目录下寻找同名 md，若仍未找到则报错
        same_name = list(latest_dir.rglob(f"{doc_name}.md"))
        if same_name:
            md_path = same_name[0]
        else:
            candidates = list(latest_dir.rglob("*.md"))
            if candidates:
                md_path = candidates[0]
            else:
                raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 Markdown 文件")
    return md_path


def _find_latest_content_list(doc_name: str) -> Path:
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    content_path = latest_dir / doc_name / f"{doc_name}_content_list.json"
    if not content_path.exists():
        # 回退：寻找同名后缀文件
        candidates = list(latest_dir.rglob(f"{doc_name}_content_list.json"))
        if candidates:
            content_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}_content_list.json")
    return content_path


def _find_latest_markdown_path(doc_name: str) -> Path:
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    md_path = latest_dir / doc_name / f"{doc_name}.md"
    if not md_path.exists():
        candidates = list(latest_dir.rglob(f"{doc_name}.md"))
        if candidates:
            md_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}.md")
    return md_path


def _extract_node_from_line(line: str, last_heading_level: int):
    raw = line.rstrip("\n")
    if not raw.strip():
        return None
    expanded = raw.expandtabs(4)
    stripped = expanded.strip()

    heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
    if heading:
        level = len(heading.group(1))
        text = heading.group(2).strip()
        return {"kind": "node", "level": level, "title": text, "content": text, "is_heading": True}

    zh_num = re.match(r"^([一二三四五六七八九十百千]+)[、.．]\s*(.*)$", stripped)
    if zh_num:
        text = zh_num.group(2).strip() or zh_num.group(1)
        return {"kind": "node", "level": 1, "title": text, "content": text, "is_heading": True}

    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)、])?\s+(.*)$", stripped)
    if numbered:
        num = numbered.group(1)
        text = numbered.group(2).strip() or num
        level = max(1, len(num.split(".")))
        return {"kind": "node", "level": level, "title": text, "content": text, "is_heading": True}

    # 非标题内容（段落、列表等）不作为模块节点，直接并入最近标题的 content
    return {"kind": "text", "text": stripped}


def _parse_markdown_to_tree(md_text: str, root_title: str) -> dict:
    root = {
        "id": "0",
        "level": 0,
        "title": root_title,
        "content": root_title,
        "order": 0,
        "children": [],
    }
    stack: List[Tuple[int, dict]] = [(0, root)]
    last_heading_level = 0
    counter = 1

    for line in md_text.splitlines():
        parsed = _extract_node_from_line(line, last_heading_level)
        if not parsed:
            continue

        if parsed["kind"] == "text":
            text = str(parsed.get("text", "")).strip()
            if not text:
                continue
            _, current_node = stack[-1]
            existing = str(current_node.get("content", "") or "")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            current_node["content"] = existing + text
            continue

        level = int(parsed["level"])
        title = str(parsed["title"])
        content = str(parsed.get("content", title))
        is_heading = bool(parsed.get("is_heading", False))

        while stack and stack[-1][0] >= level:
            stack.pop()
        _, parent_node = stack[-1]
        order = len(parent_node["children"]) + 1
        node = {
            "id": str(counter),
            "level": level,
            "title": title,
            "content": content,
            "order": order,
            "children": [],
        }
        counter += 1
        parent_node["children"].append(node)
        stack.append((level, node))
        if is_heading:
            last_heading_level = level

    return root


def _flatten_mindmap_nodes(root: dict) -> List[dict]:
    nodes: List[dict] = []

    def _walk(node: dict, path_titles: List[str], ancestor_chain: List[dict]):
        children = node.get("children") or []
        for child in children:
            child_title = str(child.get("title", "")).strip()
            child_path_titles = path_titles + ([child_title] if child_title else [])
            nodes.append(
                {
                    "id": child.get("id"),
                    "level": child.get("level"),
                    "title": child_title,
                    "path": " / ".join(child_path_titles),
                    "_node_ref": child,
                    "_ancestors": ancestor_chain[:],
                }
            )
            _walk(child, child_path_titles, ancestor_chain + [child])

    _walk(root, [], [])
    return nodes


def _classify_module(title: str) -> str:
    t = (title or "").strip().lower()
    if any(k in t for k in ["abstract", "摘要"]):
        return "abstract"
    if any(k in t for k in ["introduction", "引言", "简介", "背景介绍"]):
        return "introduction"
    if any(k in t for k in ["method", "methods", "approach", "方法", "方法论", "模型", "框架"]):
        return "method"
    if any(k in t for k in ["related work", "related", "prior work", "相关工作", "文献综述"]):
        return "related_work"
    if any(k in t for k in ["experiment", "experiments", "results", "实验", "结果", "实验结果", "评测"]):
        return "experiments_result"
    if any(k in t for k in ["title", "标题"]):
        return "title"
    return "module"


def _is_introduction_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(k in t for k in ["introduction", "引言", "简介"])


def _is_conclusion_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(k in t for k in ["conclusion", "conclusions", "结论", "总结"])


def _is_abstract_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(k in t for k in ["abstract", "摘要"])


def _is_tail_title(title: str) -> bool:
    t = (title or "").strip().lower()
    return any(
        k in t
        for k in [
            "references",
            "bibliography",
            "reference",
            "acknowledg",  # acknowledgements/acknowledgment
            "appendix",
            "appendices",
            "supplementary",
            "supplemental",
            "致谢",
            "参考文献",
            "引用文献",
            "附录",
        ]
    )


def _prune_tree_in_place(node: dict, should_remove: callable) -> None:
    children = node.get("children") or []
    kept = []
    for child in children:
        title = str(child.get("title", "")).strip()
        if should_remove(title):
            continue
        _prune_tree_in_place(child, should_remove)
        kept.append(child)
    node["children"] = kept


def _filter_mindmap_root_in_place(root: dict) -> None:
    top = root.get("children") or []
    intro_idx = None
    concl_idx = None
    for idx, n in enumerate(top):
        if _is_introduction_title(str(n.get("title", ""))):
            intro_idx = idx
            break
    if intro_idx is not None:
        for idx in range(intro_idx, len(top)):
            if _is_conclusion_title(str(top[idx].get("title", ""))):
                concl_idx = idx
                break

    def _remove_pred(title: str) -> bool:
        return _is_abstract_title(title) or _is_tail_title(title)

    if intro_idx is not None and concl_idx is not None and intro_idx <= concl_idx:
        root["children"] = top[intro_idx : concl_idx + 1]
        _prune_tree_in_place(root, _remove_pred)
        return

    root["children"] = [n for n in top if not _remove_pred(str(n.get("title", "")))]
    _prune_tree_in_place(root, _remove_pred)


def _truncate_passages(passages: List[str], max_chars: int, per_passage_chars: int) -> Tuple[str, List[str]]:
    clipped: List[str] = []
    remaining = max_chars
    for p in passages:
        if not p or remaining <= 0:
            continue
        s = str(p)
        s = s[:per_passage_chars]
        if len(s) > remaining:
            s = s[:remaining]
        clipped.append(s)
        remaining -= len(s)
    combined = "\n\n".join([f"[Context {i+1}]\n{p}" for i, p in enumerate(clipped)])
    return combined, clipped


def _collect_subtree_nodes(node: dict) -> List[dict]:
    nodes: List[dict] = []

    def _walk(n: dict):
        nodes.append(n)
        for c in n.get("children") or []:
            _walk(c)

    _walk(node)
    return nodes


def _build_subtree_text(node: dict, max_chars: int = 12000, per_line_chars: int = 600) -> str:
    lines: List[str] = []

    def _walk(n: dict, depth: int):
        title = str(n.get("title", "")).strip()
        content = str(n.get("content", "")).strip()
        text = content if content else title
        if title and content and content != title:
            if content.startswith(title):
                text = content
            else:
                text = f"{title}: {content}"
        indent = "  " * max(0, depth)
        if text:
            lines.append(f"{indent}- {text[:per_line_chars]}")
        for c in n.get("children") or []:
            _walk(c, depth + 1)

    _walk(node, 0)
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out


def _extract_main_number(title: str) -> Optional[str]:
    t = (title or "").strip()
    m = re.match(r"^(\d+)(?:\.\d+)*", t)
    if not m:
        return None
    return m.group(1)


def _extract_number_prefix(title: str) -> Optional[str]:
    t = (title or "").strip()
    m = re.match(r"^(\d+(?:\.\d+)*)", t)
    if not m:
        return None
    return m.group(1)


def _find_chapter_anchor_title(module: dict) -> str:
    title = module.get("title") or ""
    main_num = _extract_main_number(title)
    ancestors = module.get("_ancestors") or []

    def _title_of(node: dict) -> str:
        return str(node.get("title", "")).strip()

    if main_num:
        for node in reversed(ancestors):
            cand_title = _title_of(node)
            num_prefix = _extract_number_prefix(cand_title)
            if not num_prefix:
                continue
            if not num_prefix.startswith(main_num):
                continue
            if len(num_prefix.split(".")) == 1:
                return cand_title

    for node in reversed(ancestors):
        try:
            if int(node.get("level") or 0) == 1:
                cand_title = _title_of(node)
                if cand_title:
                    return cand_title
        except Exception:
            continue

    return title


def _select_prompt_path_by_anchor_title(anchor_title: str) -> str:
    module_type = _classify_module(anchor_title)
    mapping = {
        "title": "mind_map/title_prompts",
        "introduction": "mind_map/introduction_prompts",
        "method": "mind_map/method_prompts",
        "related_work": "mind_map/related_work_prompts",
    }
    return mapping.get(module_type, "mind_map/module_explain_prompts")


def _build_module_messages(
    module_title: str,
    module_path: str,
    module_type: str,
    context_text: str,
    anchor_title: str,
) -> Tuple[List[dict], dict]:
    system = PromptLoader.load("mind_map/system_prompts")
    user_prompt_path = _select_prompt_path_by_anchor_title(anchor_title)
    user = PromptLoader.load(user_prompt_path)

    module_context = (
        "【模块信息】\n"
        f"- 章节锚点：{anchor_title}\n"
        f"- 模块路径：{module_path}\n"
        f"- 模块标题：{module_title}\n\n"
        f"{context_text}"
    )

    template = user.template or ""
    if "<<PAPER_CONTENT>>" in template:
        user_text = template.replace("<<PAPER_CONTENT>>", module_context)
    else:
        user_text = user.render(
            module_title=module_title,
            module_path=module_path,
            module_type=module_type,
            retrieved_context=context_text,
            anchor_title=anchor_title,
        )
    messages = [
        {"role": "system", "content": system.template},
        {"role": "user", "content": user_text},
    ]
    return messages, {
        "prompt_id": user.id,
        "prompt_version": user.version,
    }


def _chunk_texts(texts: List[str], chunk_size: int, overlap: int) -> List[str]:
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


def _prefix_chunks(chunks: List[str]) -> List[str]:
    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]


def _build_rag_and_models(request: BaseRunConfig) -> Tuple[LinearRAG, LLM_Model]:
    embedding_model_path = _resolve_path(request.embedding_model, allow_missing=True)
    embedding_model = SentenceTransformer(str(embedding_model_path), device="cpu")
    llm_model = LLM_Model(request.llm_model)
    config = LinearRAGConfig(
        dataset_name=request.dataset_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model=request.spacy_model,
        working_dir=str(_resolve_path(request.working_dir)),
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


def _ensure_index_ready(rag: LinearRAG):
    if not rag.passage_embedding_store.hash_ids or not rag.entity_embedding_store.hash_ids:
        raise HTTPException(status_code=400, detail="未找到索引数据，请先调用 /index 完成索引构建。")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/index")
def run_index(request: IndexRequest):
    try:
        passages = _load_passages(request.dataset_name)
        output_dir = _build_output_dir(request.dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))
        rag, _ = _build_rag_and_models(request)

        rag.index(passages)
        dataset_dir = os.path.join(str(_resolve_path(request.working_dir)), request.dataset_name)
        response = {
            "status": "success",
            "dataset": request.dataset_name,
            "working_dir": str(_resolve_path(request.working_dir)),
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


@app.post("/qa")
def run_qa(request: QARequest):
    try:
        output_dir = _build_output_dir(request.dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))
        rag, _ = _build_rag_and_models(request)
        _ensure_index_ready(rag)
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


@app.post("/evaluate")
def run_evaluate(request: EvaluateRequest):
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


@app.post("/mindmap")
def generate_mindmap(request: MindmapRequest):
    try:
        md_path = _find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()
        tree = _parse_markdown_to_tree(md_text, Path(md_path).stem)
        _filter_mindmap_root_in_place(tree)
        return {
            "status": "success",
            "file": str(md_path),
            "root": tree,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/content/chunk")
def generate_chunks(request: ContentChunkRequest):
    try:
        content_path = _find_latest_content_list(request.doc_name)
        with open(content_path, "r", encoding="utf-8") as f_json:
            content_list = json.load(f_json)
        if not isinstance(content_list, list):
            raise HTTPException(status_code=400, detail="content_list.json 格式错误：应为数组")
        texts: List[str] = []
        for item in content_list:
            if isinstance(item, dict):
                text_val = item.get("text")
                if text_val:
                    texts.append(str(text_val))
        if not texts:
            raise HTTPException(status_code=400, detail="content_list.json 中未找到文本内容")

        chunk_size = LinearRAGConfig.chunk_token_size
        overlap = LinearRAGConfig.chunk_overlap_token_size
        chunks = _prefix_chunks(_chunk_texts(texts, chunk_size, overlap))

        target_dir = BASE_DIR / "data" / request.doc_name
        target_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = target_dir / "chunk.json"
        with open(chunk_path, "w", encoding="utf-8") as f_out:
            json.dump(chunks, f_out, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "content_list": str(content_path),
            "chunk_path": str(chunk_path),
            "chunk_count": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/markdown/chunk")
def generate_markdown_chunks(request: MarkdownChunkRequest):
    try:
        md_path = _find_latest_markdown_path(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        chunk_size = LinearRAGConfig.chunk_token_size
        overlap = LinearRAGConfig.chunk_overlap_token_size
        chunks = _prefix_chunks(_chunk_texts([md_text], chunk_size, overlap))

        target_dir = BASE_DIR / "dataset" / request.doc_name
        target_dir.mkdir(parents=True, exist_ok=True)
        chunk_path = target_dir / "chunks.json"
        with open(chunk_path, "w", encoding="utf-8") as f_out:
            json.dump(chunks, f_out, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "markdown": str(md_path),
            "chunk_path": str(chunk_path),
            "chunk_count": len(chunks),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mindmap/explain", response_model=MindmapExplainResponse)
def explain_mindmap_modules(request: MindmapExplainRequest):
    try:
        md_path = _find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        root = _parse_markdown_to_tree(md_text, Path(md_path).stem)
        _filter_mindmap_root_in_place(root)
        modules = _flatten_mindmap_nodes(root)
        dataset_name = request.dataset_name or request.doc_name
        if not modules:
            return {
                "status": "success",
                "doc_name": request.doc_name,
                "dataset_name": dataset_name,
                "markdown_path": str(md_path),
                "module_count": 0,
                "results": [],
                "root": root if request.include_tree else None,
                "log_path": None,
            }
        output_dir = _build_output_dir(dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))

        embedding_model_path = _resolve_path(request.embedding_model, allow_missing=True)
        embedding_model = SentenceTransformer(str(embedding_model_path), device="cpu")
        llm_model = LLM_Model(request.llm_model)
        config = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            spacy_model=request.spacy_model,
            working_dir=str(_resolve_path(request.working_dir)),
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
        rag = LinearRAG(global_config=config)
        _ensure_index_ready(rag)

        queries: List[dict] = []
        for m in modules:
            title = m["title"] or ""
            if request.include_breadcrumb_in_query and m.get("path"):
                query = f"{m['path']}\n\n{title}"
            else:
                query = title
            queries.append({"question": query, "answer": None})

        retrieval_results = rag.retrieve(queries)

        for m, r in zip(modules, retrieval_results):
            passages = (r.get("sorted_passage") or [])[: request.retrieval_top_k]
            context_text, context_clipped = _truncate_passages(
                passages,
                max_chars=request.context_max_chars,
                per_passage_chars=request.context_per_passage_chars,
            )
            m["_context_text"] = context_text
            m["_context_clipped"] = context_clipped

        id_to_module = {str(m.get("id", "")): m for m in modules if m.get("id") is not None}

        def _run_one(module: dict) -> dict:
            anchor_title = _find_chapter_anchor_title(module)
            module_type = _classify_module(anchor_title)
            node_ref = module.get("_node_ref") or {}
            subtree_nodes = _collect_subtree_nodes(node_ref) if isinstance(node_ref, dict) else []
            subtree_ids = [str(n.get("id", "")) for n in subtree_nodes if n.get("id") is not None]

            aggregated_passages: List[str] = []
            seen = set()
            for sid in subtree_ids:
                m2 = id_to_module.get(sid)
                if not m2:
                    continue
                for p in m2.get("_context_clipped") or []:
                    if p in seen:
                        continue
                    seen.add(p)
                    aggregated_passages.append(p)

            agg_context_text, agg_context_clipped = _truncate_passages(
                aggregated_passages,
                max_chars=request.context_max_chars,
                per_passage_chars=request.context_per_passage_chars,
            )
            subtree_text = _build_subtree_text(node_ref, max_chars=request.context_max_chars)
            combined_input = (
                "【模块子树内容】\n"
                f"{subtree_text}\n\n"
                "【子树检索上下文】\n"
                f"{agg_context_text}"
            ).strip()

            messages, prompt_meta = _build_module_messages(
                module_title=module.get("title", ""),
                module_path=module.get("path", ""),
                module_type=module_type,
                context_text=combined_input,
                anchor_title=anchor_title,
            )
            answer = rag.llm_model.infer(messages)
            return {
                "llm_answer": answer,
                "module_type": module_type,
                "anchor_title": anchor_title,
                "subtree_node_count": len(subtree_nodes),
                "context_clipped": agg_context_clipped,
                **prompt_meta,
            }

        results: List[dict] = []
        had_errors = False
        with ThreadPoolExecutor(max_workers=request.module_max_workers) as executor:
            future_to_module = {executor.submit(_run_one, m): m for m in modules}
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                result_item = {
                    "id": str(module.get("id", "")),
                    "title": str(module.get("title", "")),
                    "level": int(module.get("level") or 0),
                    "path": str(module.get("path", "")),
                }
                try:
                    out = future.result()
                    result_item.update(
                        {
                            "prompt_id": out.get("prompt_id"),
                            "prompt_version": out.get("prompt_version"),
                            "module_type": out.get("module_type"),
                            "anchor_title": out.get("anchor_title"),
                            "subtree_node_count": out.get("subtree_node_count"),
                            "llm_answer": out.get("llm_answer"),
                        }
                    )
                    if request.include_context:
                        result_item["context"] = out.get("context_clipped") or []
                    if request.include_tree and module.get("_node_ref") is not None:
                        node_ref = module["_node_ref"]
                        node_ref["llm_answer"] = out.get("llm_answer")
                        node_ref["module_type"] = out.get("module_type")
                        node_ref["anchor_title"] = out.get("anchor_title")
                        node_ref["subtree_node_count"] = out.get("subtree_node_count")
                        node_ref["prompt_id"] = out.get("prompt_id")
                        node_ref["prompt_version"] = out.get("prompt_version")
                        if request.include_context:
                            node_ref["context"] = result_item.get("context", [])
                except Exception as e:
                    had_errors = True
                    result_item["error"] = str(e)
                    if request.include_context:
                        result_item["context"] = module.get("_context_clipped", [])
                    if request.include_tree and module.get("_node_ref") is not None:
                        module["_node_ref"]["error"] = str(e)
                results.append(result_item)

        results.sort(key=lambda x: (int(x["id"]) if x.get("id", "").isdigit() else 10**9, str(x.get("path", ""))))
        status = "partial_success" if had_errors else "success"

        return {
            "status": status,
            "doc_name": request.doc_name,
            "dataset_name": dataset_name,
            "markdown_path": str(md_path),
            "module_count": len(modules),
            "results": results,
            "root": root if request.include_tree else None,
            "log_path": os.path.join(output_dir, "log.txt"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _safe_filename(name: str) -> str:
    return Path(name).name


def _save_mineru_json(result_json: dict, output_dir: Path) -> dict:
    output = {}
    results = result_json.get("results", {})
    for file_name, content in results.items():
        safe_name = _safe_filename(file_name)
        file_dir = output_dir / safe_name
        file_dir.mkdir(parents=True, exist_ok=True)

        def _dump_json_field(field: str, suffix: str, bucket: str):
            data = content.get(field)
            if data is None:
                return
            try:
                path = file_dir / f"{safe_name}{suffix}"
                with open(path, "w", encoding="utf-8") as f_json:
                    json.dump(data, f_json, ensure_ascii=False, indent=2)
                output.setdefault(bucket, []).append(str(path))
            except Exception:
                return

        if content.get("md_content"):
            md_path = file_dir / f"{safe_name}.md"
            with open(md_path, "w", encoding="utf-8") as f_md:
                f_md.write(content["md_content"])
            output.setdefault("md_files", []).append(str(md_path))
        _dump_json_field("middle_json", "_middle.json", "middle_json_files")
        _dump_json_field("model_output", "_model_output.json", "model_output_files")
        _dump_json_field("content_list", "_content_list.json", "content_list_files")
        if content.get("images"):
            images_dir = file_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            for img_name, data_uri in content["images"].items():
                try:
                    if "," in data_uri:
                        _, b64_data = data_uri.split(",", 1)
                    else:
                        b64_data = data_uri
                    suffix = Path(img_name).suffix or ".jpg"
                    img_path = images_dir / f"{_safe_filename(img_name)}"
                    if img_path.suffix == "":
                        img_path = img_path.with_suffix(suffix)
                    with open(img_path, "wb") as f_img:
                        f_img.write(base64.b64decode(b64_data))
                    output.setdefault("images", []).append(str(img_path))
                except Exception:
                    continue
    return output


@app.post("/mineru/parse")
def run_mineru_parse(request: MineruParseRequest):
    try:
        file_path = _resolve_path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=400, detail=f"文件不存在: {file_path}")

        # mineru_base = os.getenv("MINERU_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
        mineru_path = os.getenv("MINERU_FILE_PARSE_PATH", "/file_parse")
        mineru_path = f"/{mineru_path.lstrip('/')}"
        mineru_base = request.server_url
        if not mineru_base.startswith("http://"):
            mineru_base = f"http://{mineru_base}"
        mineru_url = f"{mineru_base}{mineru_path}"

        output_dir = _build_mineru_output_dir(request.output_dir, file_path.stem)

        form_data = {
            "backend": request.backend,
            "parse_method": request.parse_method,
            "formula_enable": str(request.formula_enable).lower(),
            "table_enable": str(request.table_enable).lower(),
            "return_md": str(request.return_md).lower(),
            "return_middle_json": str(request.return_middle_json).lower(),
            "return_model_output": str(request.return_model_output).lower(),
            "return_content_list": str(request.return_content_list).lower(),
            "return_images": str(request.return_images).lower(),
            "response_format_zip": str(request.response_format_zip).lower(),
            "start_page_id": str(request.start_page_id),
            "end_page_id": str(request.end_page_id),
        }
        # if request.lang_list:
        #     # FastAPI expects repeated keys for lists
        #     form_data["lang_list"] = request.lang_list
        if request.server_url:
            form_data["server_url"] = request.server_url

        file_mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        with open(file_path, "rb") as f:
            files = {"files": (file_path.name, f, file_mime)}
            with httpx.Client(timeout=999.0) as client:
                resp = client.post(mineru_url, data=form_data, files=files)

        if resp.status_code >= 500:
            raise HTTPException(status_code=502, detail=f"MinerU 服务错误({resp.status_code}): {resp.text}")
        if resp.status_code >= 400:
            error_text = resp.text.strip()
            hint = ""
            if resp.status_code == 404:
                hint = "（请检查 MINERU_BASE_URL/MINERU_FILE_PARSE_PATH 是否指向 MinerU 的 /file_parse 接口）"
            suffix = f": {error_text}" if error_text else ""
            raise HTTPException(status_code=400, detail=f"MinerU 请求失败({resp.status_code}){suffix}{hint}")

        if request.response_format_zip:
            zip_path = output_dir / f"{file_path.stem}_mineru.zip"
            with open(zip_path, "wb") as f_zip:
                f_zip.write(resp.content)
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(output_dir)
                    extracted_items = [str((output_dir / name).resolve()) for name in zip_ref.namelist()]
            except Exception as ex:
                raise HTTPException(status_code=502, detail=f"解压 MinerU ZIP 失败: {ex}")
            return {
                "status": "success",
                "mineru_url": mineru_url,
                "file": str(file_path),
                "output_zip": str(zip_path),
                "extracted_dir": str(output_dir),
                "extracted_items": extracted_items,
            }

        try:
            result_json = resp.json()
        except Exception:
            raise HTTPException(status_code=502, detail="MinerU 返回非 JSON 内容")

        saved = _save_mineru_json(result_json, output_dir)
        return {
            "status": "success",
            "mineru_url": mineru_url,
            "file": str(file_path),
            "output_dir": str(output_dir),
            **saved,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
