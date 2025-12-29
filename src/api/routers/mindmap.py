import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from sentence_transformers import SentenceTransformer

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.api.constants import BASE_DIR, logger
from src.api.schemas import (
    ContentChunkRequest,
    MarkdownChunkRequest,
    MarkdownTranslateRequest,
    MarkdownTranslateResponse,
    MindmapExplainRequest,
    MindmapExplainResponse,
    MindmapRequest,
)
from src.api.utils.mindmap_utils import (
    build_main_num_to_anchor_title,
    estimate_tokens_with_fallback,
    filter_mindmap_root_in_place,
    flatten_mindmap_nodes,
    mindmap_root_to_explain_markdown,
    parse_markdown_to_tree,
    prepare_module_explain_inputs,
    translate_mindmap_titles_in_place,
    truncate_passages,
)
from src.api.utils.markdown_translate_utils import translate_markdown_text
from src.api.utils.path_utils import (
    build_output_dir,
    find_latest_content_list,
    find_latest_markdown,
    find_latest_markdown_path,
    resolve_path,
)
from src.api.utils.rag_utils import (
    chunk_texts,
    ensure_index_ready,
    prefix_chunks,
)
from src.model_client import create_openai_client, get_default_timeout_s
from src.utils import LLM_Model, setup_logging

router = APIRouter()


@router.post("/mindmap")
def generate_mindmap(request: MindmapRequest):
    """生成文档的思维导图树结构。"""
    try:
        md_path = find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()
        tree = parse_markdown_to_tree(md_text, Path(md_path).stem)
        filter_mindmap_root_in_place(tree)
        return {
            "status": "success",
            "file": str(md_path),
            "root": tree,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/content/chunk")
def generate_chunks(request: ContentChunkRequest):
    """从内容列表生成 chunk 并保存到 data 目录。"""
    try:
        content_path = find_latest_content_list(request.doc_name)
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
        chunks = prefix_chunks(chunk_texts(texts, chunk_size, overlap))

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


@router.post("/markdown/chunk")
def generate_markdown_chunks(request: MarkdownChunkRequest):
    """按 Markdown 文档生成 chunk 并保存到 dataset。"""
    try:
        md_path = find_latest_markdown_path(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        chunk_size = LinearRAGConfig.chunk_token_size
        overlap = LinearRAGConfig.chunk_overlap_token_size
        chunks = prefix_chunks(chunk_texts([md_text], chunk_size, overlap))

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


@router.post("/markdown/translate", response_model=MarkdownTranslateResponse)
def translate_markdown(request: MarkdownTranslateRequest):
    """翻译 MinerU Markdown 并写入同目录。"""
    try:
        md_path = find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        llm_model = LLM_Model(request.llm_model)
        translated = translate_markdown_text(
            md_text,
            llm_model=llm_model,
            max_workers=request.max_workers,
            chunk_max_chars=request.chunk_max_chars,
        )

        output_path = md_path.with_name(f"{md_path.stem}_translate{md_path.suffix}")
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(translated)

        return {
            "status": "success",
            "doc_name": request.doc_name,
            "markdown_path": str(md_path),
            "translated_path": str(output_path),
        }
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="未找到 MinerU Markdown，请先完成 MinerU 解析。")
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mindmap/explain", response_model=MindmapExplainResponse)
def explain_mindmap_modules(request: MindmapExplainRequest):
    """解释思维导图模块内容并返回答案。"""
    try:
        md_path = find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        root = parse_markdown_to_tree(md_text, Path(md_path).stem)
        filter_mindmap_root_in_place(root)
        modules = flatten_mindmap_nodes(root)
        dataset_name = request.dataset_name or request.doc_name
        if not modules:
            return {
                "status": "success",
                "doc_name": request.doc_name,
                "dataset_name": dataset_name,
                "markdown_path": str(md_path),
                "explain_markdown_path": None,
                "explain_markdown": None,
                "module_count": 0,
                "results": [],
                "root": root if request.include_tree else None,
                "log_path": None,
            }
        output_dir = build_output_dir(dataset_name)
        setup_logging(os.path.join(output_dir, "log.txt"))
        use_batch = request.use_batch
        batch_completion_window = request.batch_completion_window
        batch_poll_interval_s = request.batch_poll_interval_s

        embedding_model_path = resolve_path(request.embedding_model, allow_missing=True)
        embedding_model = SentenceTransformer(str(embedding_model_path), device="cpu")
        llm_model = LLM_Model(request.llm_model)
        config = LinearRAGConfig(
            dataset_name=dataset_name,
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
        rag = LinearRAG(global_config=config)
        ensure_index_ready(rag)

        token_total_input = 0
        token_total_output = 0
        token_total = 0
        estimated_modules = 0

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
            context_text, context_clipped = truncate_passages(
                passages,
                max_chars=request.context_max_chars,
                per_passage_chars=request.context_per_passage_chars,
            )
            m["_context_text"] = context_text
            m["_context_clipped"] = context_clipped

        id_to_module = {str(m.get("id", "")): m for m in modules if m.get("id") is not None}
        main_num_to_anchor_title = build_main_num_to_anchor_title(modules)

        def _run_one(module: dict) -> dict:
            prep = prepare_module_explain_inputs(
                module=module,
                id_to_module=id_to_module,
                main_num_to_anchor_title=main_num_to_anchor_title,
                request=request,
            )
            response = rag.llm_model.generate(prep["messages"])
            answer = response.content

            prompt_tokens, completion_tokens, total_tokens, estimated = estimate_tokens_with_fallback(
                prep["messages"],
                answer,
                response.usage,
            )

            return {
                "llm_answer": answer,
                "module_type": prep["module_type"],
                "anchor_title": prep["anchor_title"],
                "subtree_node_count": prep["subtree_node_count"],
                "context_clipped": prep["context_clipped"],
                "token_input": prompt_tokens,
                "token_output": completion_tokens,
                "token_total": total_tokens,
                "token_estimated": estimated,
                **prep["prompt_meta"],
            }

        results: List[dict] = []
        had_errors = False
        if use_batch:
            batch_input_path = os.path.join(output_dir, "mindmap_explain_batch_input.jsonl")
            batch_output_path = os.path.join(output_dir, "mindmap_explain_batch_output.jsonl")
            module_payloads: dict = {}
            batch_lines: List[dict] = []
            for idx, module in enumerate(modules):
                prep = prepare_module_explain_inputs(
                    module=module,
                    id_to_module=id_to_module,
                    main_num_to_anchor_title=main_num_to_anchor_title,
                    request=request,
                )
                custom_id = str(module.get("id", "")) if module.get("id") is not None else ""
                if not custom_id:
                    custom_id = f"idx-{idx}"
                if custom_id in module_payloads:
                    custom_id = f"{custom_id}-{idx}"
                module_payloads[custom_id] = {
                    "module": module,
                    **prep,
                }
                batch_lines.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": request.llm_model,
                            "messages": prep["messages"],
                            "temperature": 0,
                            "max_tokens": 2000,
                        },
                    }
                )

            with open(batch_input_path, "w", encoding="utf-8") as f_out:
                for line in batch_lines:
                    f_out.write(json.dumps(line, ensure_ascii=False))
                    f_out.write("\n")

            batch_timeout_s = get_default_timeout_s()
            client, http_client = create_openai_client(timeout_s=batch_timeout_s)
            output_bytes = b""
            try:
                with open(batch_input_path, "rb") as f_in:
                    input_file = client.files.create(file=f_in, purpose="batch")
                batch = client.batches.create(
                    input_file_id=input_file.id,
                    endpoint="/v1/chat/completions",
                    completion_window=batch_completion_window,
                )
                while True:
                    batch_status = client.batches.retrieve(batch.id)
                    status = getattr(batch_status, "status", None)
                    if status == "completed":
                        output_file_id = batch_status.output_file_id
                        output = client.files.content(output_file_id)
                        output_bytes = output.read()
                        break
                    if status in {"failed", "expired", "cancelled"}:
                        raise RuntimeError(f"Batch task {status}")
                    time.sleep(batch_poll_interval_s)
            finally:
                http_client.close()

            with open(batch_output_path, "wb") as f_out:
                f_out.write(output_bytes)

            output_text = output_bytes.decode("utf-8", errors="replace")
            batch_outputs: dict = {}
            for line in output_text.splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                custom_id = str(item.get("custom_id", ""))
                if custom_id:
                    batch_outputs[custom_id] = item

            for custom_id, payload in module_payloads.items():
                module = payload["module"]
                result_item = {
                    "id": str(module.get("id", "")),
                    "title": str(module.get("title", "")),
                    "level": int(module.get("level") or 0),
                    "path": str(module.get("path", "")),
                }
                output_item = batch_outputs.get(custom_id)
                if not output_item:
                    had_errors = True
                    result_item["error"] = f"batch output missing for {custom_id}"
                    if request.include_context:
                        result_item["context"] = payload.get("context_clipped") or []
                    if request.include_tree and module.get("_node_ref") is not None:
                        module["_node_ref"]["error"] = result_item["error"]
                    results.append(result_item)
                    continue

                if output_item.get("error"):
                    had_errors = True
                    result_item["error"] = str(output_item.get("error"))
                    if request.include_context:
                        result_item["context"] = payload.get("context_clipped") or []
                    if request.include_tree and module.get("_node_ref") is not None:
                        module["_node_ref"]["error"] = result_item["error"]
                    results.append(result_item)
                    continue

                response_obj = output_item.get("response") or {}
                status_code = response_obj.get("status_code")
                body = response_obj.get("body") or {}
                if status_code is not None and int(status_code) != 200:
                    had_errors = True
                    result_item["error"] = f"batch status {status_code}"
                    if request.include_context:
                        result_item["context"] = payload.get("context_clipped") or []
                    if request.include_tree and module.get("_node_ref") is not None:
                        module["_node_ref"]["error"] = result_item["error"]
                    results.append(result_item)
                    continue

                answer = ""
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    answer = message.get("content") or ""

                prompt_tokens, completion_tokens, total_tokens, estimated = estimate_tokens_with_fallback(
                    payload.get("messages") or [],
                    answer,
                    body.get("usage") or {},
                )

                token_total_input += prompt_tokens
                token_total_output += completion_tokens
                token_total += total_tokens
                if estimated:
                    estimated_modules += 1

                logger.info(
                    "mindmap_explain module_id=%s path=%s anchor=%s input_tokens=%s output_tokens=%s total_tokens=%s estimated=%s",
                    result_item.get("id"),
                    result_item.get("path"),
                    payload.get("anchor_title"),
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    estimated,
                )
                result_item.update(
                    {
                        "prompt_id": payload.get("prompt_meta", {}).get("prompt_id"),
                        "prompt_version": payload.get("prompt_meta", {}).get("prompt_version"),
                        "module_type": payload.get("module_type"),
                        "anchor_title": payload.get("anchor_title"),
                        "subtree_node_count": payload.get("subtree_node_count"),
                        "llm_answer": answer,
                    }
                )
                if request.include_context:
                    result_item["context"] = payload.get("context_clipped") or []
                if request.include_tree and module.get("_node_ref") is not None:
                    node_ref = module["_node_ref"]
                    node_ref["llm_answer"] = answer
                    node_ref["module_type"] = payload.get("module_type")
                    node_ref["anchor_title"] = payload.get("anchor_title")
                    node_ref["subtree_node_count"] = payload.get("subtree_node_count")
                    node_ref["prompt_id"] = payload.get("prompt_meta", {}).get("prompt_id")
                    node_ref["prompt_version"] = payload.get("prompt_meta", {}).get("prompt_version")
                    if request.include_context:
                        node_ref["context"] = result_item.get("context", [])
                results.append(result_item)
        else:
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
                        token_total_input += int(out.get("token_input") or 0)
                        token_total_output += int(out.get("token_output") or 0)
                        token_total += int(out.get("token_total") or 0)
                        if out.get("token_estimated"):
                            estimated_modules += 1

                        logger.info(
                            "mindmap_explain module_id=%s path=%s anchor=%s input_tokens=%s output_tokens=%s total_tokens=%s estimated=%s",
                            result_item.get("id"),
                            result_item.get("path"),
                            out.get("anchor_title"),
                            out.get("token_input"),
                            out.get("token_output"),
                            out.get("token_total"),
                            out.get("token_estimated"),
                        )
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

        logger.info(
            "mindmap_explain_total modules=%s input_tokens=%s output_tokens=%s total_tokens=%s estimated_modules=%s",
            len(modules),
            token_total_input,
            token_total_output,
            token_total,
            estimated_modules,
        )

        results.sort(key=lambda x: (int(x["id"]) if x.get("id", "").isdigit() else 10**9, str(x.get("path", ""))))
        status = "partial_success" if had_errors else "success"

        explain_md_path = None
        explain_md = None
        if request.include_tree and isinstance(root, dict):
            translate_mindmap_titles_in_place(root, rag.llm_model, max_workers=request.module_max_workers)
            explain_md = mindmap_root_to_explain_markdown(root)
            explain_md_path = os.path.join(output_dir, "mindmap_explain.md")
            with open(explain_md_path, "w", encoding="utf-8") as f_md_out:
                f_md_out.write(explain_md)

        return {
            "status": status,
            "doc_name": request.doc_name,
            "dataset_name": dataset_name,
            "markdown_path": str(md_path),
            "explain_markdown_path": explain_md_path,
            "explain_markdown": explain_md,
            "module_count": len(modules),
            "results": results,
            "root": root if request.include_tree else None,
            "log_path": os.path.join(output_dir, "log.txt"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
