import json
import os
import subprocess
import time
import uuid
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
    MarkdownTranslateWithImageRequest,
    MarkdownTranslateWithImageResponse,
    MarkdownToDocxRequest,
    MarkdownToDocxResponse,
    MarkdownAssetAnalyzeRequest,
    MarkdownAssetAnalyzeResponse,
    MindmapExplainRequest,
    MindmapExplainResponse,
    MindmapRequest,
)
from src.api.utils.markdown_asset_analyze_utils import (
    build_asset_messages,
    build_asset_query,
    extract_local_context,
    load_image_data_url,
    parse_markdown_image,
    resolve_asset_image,
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
from src.api.utils.markdown_translate_utils import (
    PlaceholderStore,
    REFERENCE_HEADING_RE,
    translate_html_table,
    translate_markdown_text,
    translate_text,
)
from src.api.utils.markdown_image_translate_utils import translate_markdown_images
from src.api.utils.html_table_converter import preprocess_markdown_for_docx
from src.api.utils.path_utils import (
    build_output_dir,
    find_doc_directory,
    find_latest_content_list,
    find_latest_markdown,
    find_latest_markdown_path,
    find_latest_middle_json,
    find_latest_translated_markdown,
    find_latest_translated_with_image_markdown,
    resolve_path,
)
from src.api.utils.rag_utils import (
    chunk_texts,
    ensure_index_ready,
    prefix_chunks,
)
from src.model_client import create_openai_client, get_default_timeout_s
from src.utils import LLM_Model, setup_logging
from src.api.markdown_to_mindmap import convert_markdown_to_mindmap

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
    """翻译 MinerU middle.json 并写入同目录。"""
    try:
        middle_path = find_latest_middle_json(request.doc_name)
        with open(middle_path, "r", encoding="utf-8") as f_json:
            middle_data = json.load(f_json)

        pdf_info = middle_data.get("pdf_info") or []
        blocks_to_translate = []

        def _iter_lines(block: dict):
            for line in block.get("lines") or []:
                yield line
            for sub_block in block.get("blocks") or []:
                for line in sub_block.get("lines") or []:
                    yield line

        def _iter_spans(block: dict):
            for line in _iter_lines(block):
                for span in line.get("spans") or []:
                    yield span

        def _build_block_text(block: dict) -> tuple[str, PlaceholderStore]:
            store = PlaceholderStore(prefix="FORMULA", items=[])
            parts = []
            block_type = block.get("type")
            for span in _iter_spans(block):
                content = span.get("content")
                if content is None:
                    continue
                span_type = span.get("type")
                if span_type == "inline_equation":
                    content = store.add(f"${content}$")
                elif span_type in ("equation", "interline_equation"):
                    if block_type not in ("equation", "interline_equation"):
                        content = store.add(f"${content}$")
                    else:
                        content = store.add(content)
                parts.append(content)
            return "".join(parts), store

        def _build_ref_text_content(block: dict) -> str:
            lines = []
            for line in _iter_lines(block):
                parts = []
                for span in line.get("spans") or []:
                    content = span.get("content")
                    if content is None:
                        continue
                    parts.append(content)
                if parts:
                    lines.append("".join(parts).strip())
            cleaned = [line for line in lines if line]
            if not cleaned:
                return ""
            return "  \n".join(cleaned) + "  "

        def _extract_table_html(block: dict) -> str:
            for span in _iter_spans(block):
                html = span.get("html")
                if html:
                    return html
            return ""

        in_references = False

        def _is_reference_heading(block: dict) -> bool:
            parts = []
            for span in _iter_spans(block):
                if span.get("type") != "text":
                    continue
                content = span.get("content")
                if content:
                    parts.append(content)
            text = "".join(parts).strip()
            if not text:
                return False
            return REFERENCE_HEADING_RE.match(text) is not None

        for page in pdf_info:
            for block in page.get("para_blocks") or []:
                if not in_references and _is_reference_heading(block):
                    in_references = True
                has_ref_text = any(
                    (sub_block or {}).get("type") == "ref_text"
                    for sub_block in (block.get("blocks") or [])
                )
                block_type = block.get("type")
                if block_type == "ref_text":
                    block_text = _build_ref_text_content(block)
                    store = PlaceholderStore(prefix="FORMULA", items=[])
                else:
                    block_text, store = _build_block_text(block)
                table_html = _extract_table_html(block) if block.get("type") == "table" else ""
                skip_translate = in_references or has_ref_text or block.get("type") == "ref_text"
                blocks_to_translate.append((block, block_text, store, table_html, skip_translate))

        llm_model = LLM_Model(request.llm_model)

        def _translate_block(
            block_type: str,
            text: str,
            store: PlaceholderStore,
            table_html: str,
            skip_translate: bool,
        ) -> str:
            if skip_translate:
                original_text = store.restore(text) if store.items else text
                if block_type == "table" and table_html:
                    if not original_text.strip():
                        return table_html
                    return f"{table_html}\n\n{original_text.strip()}"
                return original_text
            if block_type == "table" and table_html:
                translated_table = translate_html_table(table_html, lambda t: translate_text(t, llm_model))
                if not text.strip():
                    return translated_table
                translated_caption = translate_markdown_text(
                    text,
                    llm_model=llm_model,
                    max_workers=1,
                    chunk_max_chars=request.chunk_max_chars,
                )
                translated_caption = store.restore(translated_caption) if store.items else translated_caption
                return f"{translated_table}\n\n{translated_caption.strip()}"
            if not text.strip():
                return text
            if block_type in ("equation", "interline_equation"):
                return store.restore(text) if store.items else text
            translated = translate_markdown_text(
                text,
                llm_model=llm_model,
                max_workers=1,
                chunk_max_chars=request.chunk_max_chars,
            )
            return store.restore(translated) if store.items else translated

        translated_results = [None] * len(blocks_to_translate)
        with ThreadPoolExecutor(max_workers=request.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _translate_block,
                    block.get("type"),
                    text,
                    store,
                    table_html,
                    skip_translate,
                ): idx
                for idx, (block, text, store, table_html, skip_translate) in enumerate(blocks_to_translate)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                translated_results[idx] = future.result()

        title_count = 0
        for (block, _, _, _, _), translated in zip(blocks_to_translate, translated_results):
            if block.get("type") == "title":
                heading = "#" if title_count == 0 else "##"
                title_count += 1
                content = translated.strip()
                block["translate_content"] = f"{heading} {content}" if content else translated
            else:
                block["translate_content"] = translated

        middle_translate_path = middle_path.with_name(f"{middle_path.stem}_translate{middle_path.suffix}")
        with open(middle_translate_path, "w", encoding="utf-8") as f_out:
            json.dump(middle_data, f_out, ensure_ascii=False, indent=2)

        def _process_block_translate_content(block: dict):
            """处理单个块的 translate_content，返回要添加到 markdown 的内容列表"""
            results = []
            block_type = block.get("type")
            block_sub_type = block.get("sub_type")
            translate_content = block.get("translate_content") or ""
            
            has_ref_text = any(
                (sub_block or {}).get("type") == "ref_text"
                for sub_block in (block.get("blocks") or [])
            )
            
            # 处理顶层块的 translate_content
            if block_type == "title":
                if translate_content.strip():
                    results.append(translate_content.strip())
            elif block_type == "ref_text":
                ref_text = _build_ref_text_content(block)
                if ref_text.strip():
                    results.append(ref_text.strip())
            elif block_type == "text":
                if translate_content.strip():
                    results.append(translate_content.strip())
            elif block_type == "image":
                image_path = None
                for span in _iter_spans(block):
                    image_path = span.get("image_path")
                    if image_path:
                        break
                if image_path:
                    results.append(f"![](images/{image_path})")
            elif block_type == "table":
                if translate_content.strip():
                    results.append(translate_content.strip())
            elif block_type in ("equation", "interline_equation"):
                if translate_content.strip():
                    results.append(f"$${translate_content.strip()}$$")
            elif block_type == "code_body":
                # 处理代码块，特别是算法伪代码
                if translate_content.strip():
                    # 算法块使用代码块格式
                    if block_sub_type == "algorithm":
                        results.append(f"```\n{translate_content.strip()}\n```")
                    else:
                        # 其他代码块类型也使用代码块格式
                        results.append(f"```\n{translate_content.strip()}\n```")
            elif has_ref_text:
                ref_text = _build_ref_text_content(block)
                if ref_text.strip():
                    results.append(ref_text.strip())
            else:
                # 如果顶层块没有处理，但有 translate_content，也添加
                if translate_content.strip():
                    results.append(translate_content.strip())
            
            # 处理内部的 blocks 字段（子块）
            for sub_block in block.get("blocks") or []:
                sub_translate_content = sub_block.get("translate_content") or ""
                if sub_translate_content.strip():
                    # 递归处理子块
                    sub_results = _process_block_translate_content(sub_block)
                    results.extend(sub_results)
            
            return results

        markdown_blocks = []
        for page in pdf_info:
            for block in page.get("para_blocks") or []:
                block_contents = _process_block_translate_content(block)
                markdown_blocks.extend(block_contents)
        markdown_text = "\n\n".join(markdown_blocks)

        base_stem = middle_path.stem[:-7] if middle_path.stem.endswith("_middle") else middle_path.stem
        output_path = middle_path.with_name(f"{base_stem}_translate.md")
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(markdown_text)

        return {
            "status": "success",
            "doc_name": request.doc_name,
            "markdown_path": str(middle_path),
            "translated_path": str(output_path),
            "middle_translate_path": str(middle_translate_path),
        }
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="未找到 MinerU middle.json，请先完成 MinerU 解析。")
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/markdown/translate_with_image", response_model=MarkdownTranslateWithImageResponse)
def translate_markdown_with_image(request: MarkdownTranslateWithImageRequest):
    """翻译 Markdown 图片文本并替换图片链接。"""
    try:
        md_path = find_latest_translated_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        llm_model = LLM_Model(request.llm_model)
        translated_md, output_images = translate_markdown_images(
            md_text,
            md_path=md_path,
            llm_model=llm_model,
            ocr_model=request.ocr_model,
        )

        output_path = md_path.with_name(f"{md_path.stem}_translate_with_image{md_path.suffix}")
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(translated_md)

        return {
            "status": "success",
            "doc_name": request.doc_name,
            "markdown_path": str(md_path),
            "translated_path": str(output_path),
            "image_count": len(output_images),
        }
    except HTTPException as exc:
        if exc.status_code == 404:
            raise HTTPException(status_code=404, detail="未找到翻译后的 Markdown，请先完成 Markdown 翻译。")
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/markdown/to_docx", response_model=MarkdownToDocxResponse)
def convert_markdown_to_docx(request: MarkdownToDocxRequest):
    """将 Markdown 文件转换为 DOCX 格式。"""
    try:
        # 查找 _translate_with_image.md 文件
        md_path = find_latest_translated_with_image_markdown(request.doc_name)
        logger.info("找到 Markdown 文件: %s", md_path)
        
        # 验证文件存在
        if not md_path.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Markdown 文件不存在: {md_path}"
            )
        
        # 读取原始 Markdown 内容
        with open(md_path, "r", encoding="utf-8") as f:
            original_md = f.read()
        
        logger.info("读取 Markdown 文件，大小: %d 字符", len(original_md))
        
        # 预处理 Markdown（转换 HTML 表格为 Markdown 表格）
        processed_md, stats = preprocess_markdown_for_docx(original_md)
        logger.info(
            "预处理完成 - HTML 表格转换: %d 个，原始大小: %d，处理后大小: %d",
            stats['html_tables_converted'],
            stats['original_length'],
            stats['processed_length']
        )
        
        # 创建临时文件保存处理后的 Markdown
        temp_md_path = md_path.with_name(f"{md_path.stem}_temp.md")
        with open(temp_md_path, "w", encoding="utf-8") as f:
            f.write(processed_md)
        
        logger.info("创建临时文件: %s", temp_md_path)
        
        # 生成输出路径（同级目录）
        docx_path = md_path.with_suffix(".docx")
        logger.info("目标 DOCX 路径: %s", docx_path)
        
        # 检查 pandoc 是否可用
        try:
            pandoc_version = subprocess.run(
                ["pandoc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("使用 pandoc 版本: %s", pandoc_version.stdout.split('\n')[0])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("pandoc 不可用: %s", str(e))
            # 清理临时文件
            if temp_md_path.exists():
                temp_md_path.unlink()
            raise HTTPException(
                status_code=500,
                detail="pandoc 未安装或不可用，请先安装 pandoc (https://pandoc.org/installing.html)"
            )
        
        # 构建 pandoc 命令
        # 使用 --wrap=preserve 保留换行
        # 使用 --extract-media 提取图片到指定目录
        media_dir = md_path.parent / "media"
        
        pandoc_cmd = [
            "pandoc",
            str(temp_md_path),
            "-o", str(docx_path),
            "--wrap=preserve",
            f"--extract-media={media_dir}",
            "--from=markdown",
            "--to=docx"
        ]
        
        logger.info("执行 pandoc 命令: %s", " ".join(pandoc_cmd))
        
        # 执行转换
        try:
            result = subprocess.run(
                pandoc_cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=str(md_path.parent)
            )
            
            if result.stdout:
                logger.info("pandoc 输出: %s", result.stdout)
            if result.stderr:
                logger.warning("pandoc 警告: %s", result.stderr)
                
        except subprocess.CalledProcessError as e:
            logger.error("pandoc 转换失败: %s", e.stderr)
            # 清理临时文件
            if temp_md_path.exists():
                temp_md_path.unlink()
            raise HTTPException(
                status_code=500,
                detail=f"pandoc 转换失败: {e.stderr}"
            )
        finally:
            # 清理临时文件
            if temp_md_path.exists():
                temp_md_path.unlink()
                logger.info("已删除临时文件: %s", temp_md_path)
        
        # 验证输出文件
        if not docx_path.exists():
            raise HTTPException(
                status_code=500,
                detail="DOCX 文件生成失败"
            )
        
        file_size = docx_path.stat().st_size
        logger.info("转换成功，DOCX 文件大小: %d 字节", file_size)
        
        return {
            "status": "success",
            "doc_name": request.doc_name,
            "markdown_path": str(md_path),
            "docx_path": str(docx_path),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("转换过程中发生错误: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/markdown/asset/analyze", response_model=MarkdownAssetAnalyzeResponse)
def analyze_markdown_asset(request: MarkdownAssetAnalyzeRequest):
    """分析 Markdown 资源（图片/表格/公式）。"""
    try:
        dataset_name = request.dataset_name or request.doc_name
        output_dir = build_output_dir(dataset_name)
        log_path = os.path.join(output_dir, "markdown_asset_analyze.log")
        setup_logging(log_path)
        request_id = uuid.uuid4().hex[:8]

        md_path = find_latest_markdown(request.doc_name)
        with open(md_path, "r", encoding="utf-8") as f_md:
            md_text = f_md.read()

        alt_text, image_ref, title_text = parse_markdown_image(request.asset_markdown)
        query = build_asset_query(alt_text, title_text, image_ref)

        local_context = extract_local_context(
            md_text,
            request.asset_markdown,
            image_ref,
            request.local_context_window_chars,
        )

        image_path, image_url = resolve_asset_image(md_path, image_ref)
        if image_path:
            image_url = load_image_data_url(image_path)
        if not image_url:
            raise HTTPException(status_code=404, detail="未找到可用的图片地址")

        logger.info(
            "Asset analyze start request_id=%s doc=%s dataset=%s ref=%s query=%s",
            request_id,
            request.doc_name,
            dataset_name,
            image_ref,
            query,
        )
        logger.info(
            "Asset context lengths request_id=%s local=%d",
            request_id,
            len(local_context),
        )

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

        retrieval_results = rag.retrieve([{"question": query, "answer": None}])
        passages = (retrieval_results[0].get("sorted_passage") or [])[: request.retrieval_top_k]
        retrieval_context, _ = truncate_passages(
            passages,
            max_chars=request.context_max_chars,
            per_passage_chars=request.context_per_passage_chars,
        )
        logger.info(
            "Asset retrieval context request_id=%s length=%d passages=%d",
            request_id,
            len(retrieval_context),
            len(passages),
        )

        messages, prompt_meta = build_asset_messages(
            asset_markdown=request.asset_markdown,
            query=query,
            local_context=local_context,
            retrieval_context=retrieval_context,
            image_url=image_url,
        )
        response = llm_model.generate(messages)
        analysis = response.content or ""

        logger.info(
            "Asset analyze done request_id=%s output_len=%d prompt=%s@%s",
            request_id,
            len(analysis),
            prompt_meta.get("prompt_id"),
            prompt_meta.get("prompt_version"),
        )
        if analysis:
            logger.info("Asset analyze output request_id=%s preview=%s", request_id, analysis[:300])

        return {"analysis": analysis}
    except HTTPException:
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
            md_path_obj = Path(md_path)
            explain_md_path = str(
                md_path_obj.with_name(f"{md_path_obj.stem}_mindmap_explain{md_path_obj.suffix or '.md'}")
            )
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


@router.post("/mindmap/to_reactflow")
def convert_mindmap_to_reactflow(request: MindmapRequest):
    """
    将思维导图解释的 markdown 转换为 React Flow 格式
    
    Args:
        request: 包含 doc_name 的请求
        
    Returns:
        React Flow 格式的思维导图数据
    """
    try:
        # 使用项目根目录
        base_dir = str(BASE_DIR)
        
        # 转换为思维导图
        result = convert_markdown_to_mindmap(request.doc_name, base_dir)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"未找到文件: {request.doc_name}，请确保文件在 output/mineru 目录下"
            )
        
        # 查找生成的文件路径（支持文件名中包含空格）
        doc_dir = find_doc_directory(request.doc_name)
        json_file = None
        for f in doc_dir.rglob("*_mindmap_explain_mindmap.json"):
            json_file = str(f)
            break
        
        return {
            "status": "success",
            "doc_name": request.doc_name,
            "message": "思维导图生成成功",
            "file_path": json_file,
            "node_count": len(result.get("nodes", [])),
            "edge_count": len(result.get("edges", [])),
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("生成思维导图失败: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"生成思维导图失败: {str(e)}"
        )


@router.get("/mindmap/reactflow/{doc_name}")
def get_reactflow_mindmap(doc_name: str):
    """
    获取已生成的 React Flow 格式思维导图
    
    Args:
        doc_name: 文档名称（不含扩展名）
        
    Returns:
        React Flow 格式的思维导图数据
    """
    try:
        # 查找文档目录（支持文件名中包含空格）
        doc_dir = find_doc_directory(doc_name)
        
        # 查找 JSON 文件
        json_file = None
        for f in doc_dir.rglob("*_mindmap_explain_mindmap.json"):
            json_file = f
            break
        
        if json_file is None or not json_file.exists():
            raise HTTPException(
                status_code=404,
                detail=f"未找到思维导图文件: {doc_name}，请先生成思维导图"
            )
        
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "status": "success",
            "doc_name": doc_name,
            "message": "获取思维导图成功",
            "file_path": str(json_file),
            "node_count": len(data.get("nodes", [])),
            "edge_count": len(data.get("edges", [])),
            "data": data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("获取思维导图失败: %s", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"获取思维导图失败: {str(e)}"
        )
