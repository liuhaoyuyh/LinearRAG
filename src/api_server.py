import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import re
import base64
import mimetypes
import zipfile

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import httpx

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.utils import LLM_Model, setup_logging

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
        return level, text, text, True

    zh_num = re.match(r"^([一二三四五六七八九十百千]+)[、.．]\s*(.*)$", stripped)
    if zh_num:
        text = zh_num.group(2).strip() or zh_num.group(1)
        return 1, text, text, True

    numbered = re.match(r"^(\d+(?:\.\d+)*)(?:[.)、])?\s+(.*)$", stripped)
    if numbered:
        num = numbered.group(1)
        text = numbered.group(2).strip() or num
        level = max(1, len(num.split(".")))
        return level, text, text, True

    indent = len(expanded) - len(expanded.lstrip(" "))
    bullet = re.match(r"^[-*+]\s+(.*)$", stripped)
    if bullet:
        base = last_heading_level if last_heading_level > 0 else 0
        level = base + 1 + indent // 2
        text = bullet.group(1).strip()
        return max(1, level), text, text, False

    numbered_list = re.match(r"^(\d+)\.\s+(.*)$", stripped)
    if numbered_list:
        base = last_heading_level if last_heading_level > 0 else 0
        level = base + 1 + indent // 2
        text = numbered_list.group(2).strip() or numbered_list.group(1)
        return max(1, level), text, text, False

    # 普通段落挂到最近标题下
    paragraph_level = (last_heading_level if last_heading_level > 0 else 0) + 1
    return paragraph_level, stripped, stripped, False


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
        level, title, content, is_heading = parsed
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_level, parent_node = stack[-1]
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