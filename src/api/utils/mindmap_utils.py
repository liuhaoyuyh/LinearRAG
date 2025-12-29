import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from src.utils import LLM_Model, estimate_message_tokens, estimate_token_count
from prompts.loader import PromptLoader


def extract_node_from_line(line: str, last_heading_level: int):
    """解析 Markdown 单行，生成节点或文本信息。"""
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

    return {"kind": "text", "text": stripped}


def parse_markdown_to_tree(md_text: str, root_title: str) -> dict:
    """将 Markdown 内容解析为树结构。"""
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
        parsed = extract_node_from_line(line, last_heading_level)
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


def flatten_mindmap_nodes(root: dict) -> List[dict]:
    """展开树结构为模块节点列表。"""
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


def classify_module(title: str) -> str:
    """基于标题判断模块类型。"""
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


def is_introduction_title(title: str) -> bool:
    """判断标题是否是引言类。"""
    t = (title or "").strip().lower()
    return any(k in t for k in ["introduction", "引言", "简介"])


def is_conclusion_title(title: str) -> bool:
    """判断标题是否是结论类。"""
    t = (title or "").strip().lower()
    return any(k in t for k in ["conclusion", "conclusions", "结论", "总结"])


def is_abstract_title(title: str) -> bool:
    """判断标题是否是摘要类。"""
    t = (title or "").strip().lower()
    return any(k in t for k in ["abstract", "摘要"])


def is_tail_title(title: str) -> bool:
    """判断标题是否是参考/附录等尾部模块。"""
    t = (title or "").strip().lower()
    return any(
        k in t
        for k in [
            "references",
            "bibliography",
            "reference",
            "acknowledg",
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


def prune_tree_in_place(node: dict, should_remove: callable) -> None:
    """在树上按条件删除节点。"""
    children = node.get("children") or []
    kept = []
    for child in children:
        title = str(child.get("title", "")).strip()
        if should_remove(title):
            continue
        prune_tree_in_place(child, should_remove)
        kept.append(child)
    node["children"] = kept


def filter_mindmap_root_in_place(root: dict) -> None:
    """过滤头尾模块，仅保留核心结构。"""
    top = root.get("children") or []
    intro_idx = None
    concl_idx = None
    for idx, n in enumerate(top):
        if is_introduction_title(str(n.get("title", ""))):
            intro_idx = idx
            break
    if intro_idx is not None:
        for idx in range(intro_idx, len(top)):
            if is_conclusion_title(str(top[idx].get("title", ""))):
                concl_idx = idx
                break

    def _remove_pred(title: str) -> bool:
        return is_abstract_title(title) or is_tail_title(title)

    if intro_idx is not None and concl_idx is not None and intro_idx <= concl_idx:
        root["children"] = top[intro_idx: concl_idx + 1]
        prune_tree_in_place(root, _remove_pred)
        return

    root["children"] = [n for n in top if not _remove_pred(str(n.get("title", "")))]
    prune_tree_in_place(root, _remove_pred)


def truncate_passages(passages: List[str], max_chars: int, per_passage_chars: int) -> Tuple[str, List[str]]:
    """截断 passage 列表并组合成上下文字符串。"""
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


def collect_subtree_nodes(node: dict) -> List[dict]:
    """收集节点子树内所有节点。"""
    nodes: List[dict] = []

    def _walk(n: dict):
        nodes.append(n)
        for c in n.get("children") or []:
            _walk(c)

    _walk(node)
    return nodes


def build_subtree_text(node: dict, max_chars: int = 12000, per_line_chars: int = 600) -> str:
    """将子树内容拼接为文本描述。"""
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


def mindmap_root_to_explain_markdown(root: dict) -> str:
    """将带解释的树结构导出为 Markdown。"""
    lines: List[str] = []

    def _export_heading_level(node: dict) -> int:
        title = str(node.get("title_zh") or node.get("title", "")).strip()
        m = re.match(r"^\s*(\d+(?:\.\d+)*)", title)
        if m:
            segs = [s for s in m.group(1).split(".") if s.strip() != ""]
            if segs:
                return min(max(len(segs), 1), 6)
        level = int(node.get("level") or 1)
        return min(max(level, 1), 6)

    def _walk(node: dict):
        for child in node.get("children") or []:
            title = str(child.get("title_zh") or child.get("title", "")).strip()
            heading_level = _export_heading_level(child)
            if title:
                lines.append(f"{'#' * heading_level} {title}")
                lines.append("")

            answer = child.get("llm_answer")
            if isinstance(answer, str) and answer.strip():
                lines.append(answer.strip())
                lines.append("")

            _walk(child)

    if isinstance(root, dict):
        _walk(root)

    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def looks_like_already_chinese(title: str) -> bool:
    """粗略判断字符串是否已经是中文标题。"""
    t = (title or "").strip()
    if not t:
        return True
    has_cjk = re.search(r"[\u4e00-\u9fff]", t) is not None
    has_ascii_alpha = re.search(r"[A-Za-z]", t) is not None
    return has_cjk and not has_ascii_alpha


def split_number_prefix(title: str) -> Tuple[str, str]:
    """拆分标题中的编号前缀。"""
    t = (title or "").strip()
    if not t:
        return "", ""
    m = re.match(r"^(\d+(?:\.\d+)*[.)]?)\s+(.*)$", t)
    if not m:
        return "", t
    prefix = m.group(1).strip()
    rest = (m.group(2) or "").strip()
    if not rest:
        return "", t
    return f"{prefix} ", rest


def clean_single_line(text: str) -> str:
    """清理模型输出为单行内容。"""
    if not isinstance(text, str):
        text = str(text or "")
    s = text.strip()
    if not s:
        return ""
    for line in s.splitlines():
        line = line.strip()
        if line:
            s = line
            break
    s = s.strip().strip("`").strip().strip('"').strip("'").strip()
    return s


def build_title_translate_messages(title: str) -> Tuple[List[dict], dict]:
    """构造标题翻译提示词。"""
    system = PromptLoader.load("mind_map/system_prompts")
    user = PromptLoader.load("mind_map/title_translate_prompts")
    user_text = user.render(title=title)
    return (
        [
            {"role": "system", "content": system.template},
            {"role": "user", "content": user_text},
        ],
        {"prompt_id": user.id, "prompt_version": user.version},
    )


def translate_title_to_zh(llm_model: LLM_Model, title: str) -> str:
    """调用模型将标题翻译为中文。"""
    raw = str(title or "").strip()
    if not raw:
        return ""
    if looks_like_already_chinese(raw):
        return raw

    prefix, rest = split_number_prefix(raw)
    target = rest if prefix else raw
    try:
        messages, _ = build_title_translate_messages(target)
        response = llm_model.generate(messages)
        translated = clean_single_line(getattr(response, "content", ""))
        if not translated:
            return raw
        return f"{prefix}{translated}" if prefix else translated
    except Exception:
        return raw


def translate_mindmap_titles_in_place(root: dict, llm_model: LLM_Model, max_workers: int) -> None:
    """并发翻译树中标题。"""
    if not isinstance(root, dict):
        return

    unique_titles: List[str] = []
    seen = set()

    def _collect(node: dict):
        for child in node.get("children") or []:
            title = str(child.get("title", "")).strip()
            if title and title not in seen:
                seen.add(title)
                unique_titles.append(title)
            if isinstance(child, dict):
                _collect(child)

    _collect(root)
    if not unique_titles:
        return

    title_map: dict = {}
    worker_count = max(1, int(max_workers or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_title = {executor.submit(translate_title_to_zh, llm_model, t): t for t in unique_titles}
        for future in as_completed(future_to_title):
            title = future_to_title[future]
            try:
                title_map[title] = str(future.result() or "").strip() or title
            except Exception:
                title_map[title] = title

    def _apply(node: dict):
        for child in node.get("children") or []:
            title = str(child.get("title", "")).strip()
            if title:
                child["title_zh"] = title_map.get(title, title)
            if isinstance(child, dict):
                _apply(child)

    _apply(root)


def extract_main_number(title: str) -> Optional[str]:
    """提取编号主序号。"""
    t = (title or "").strip()
    m = re.match(r"^(\d+)(?:\.\d+)*", t)
    if not m:
        return None
    return m.group(1)


def extract_number_prefix(title: str) -> Optional[str]:
    """提取编号前缀。"""
    t = (title or "").strip()
    m = re.match(r"^(\d+(?:\.\d+)*)", t)
    if not m:
        return None
    return m.group(1)


def build_main_num_to_anchor_title(modules: List[dict]) -> dict:
    """构建主章节编号到标题的映射。"""
    out: dict = {}
    for m in modules or []:
        title = str(m.get("title", "")).strip()
        if not title:
            continue
        num_prefix = extract_number_prefix(title)
        if not num_prefix:
            continue
        if len([s for s in num_prefix.split(".") if s.strip() != ""]) != 1:
            continue
        main_num = num_prefix.split(".")[0].strip()
        if main_num and main_num not in out:
            out[main_num] = title
    return out


def find_chapter_anchor_title(module: dict, main_num_to_anchor_title: Optional[dict] = None) -> str:
    """根据编号或祖先节点确定章节锚点标题。"""
    title = module.get("title") or ""
    main_num = extract_main_number(title)
    ancestors = module.get("_ancestors") or []

    def _title_of(node: dict) -> str:
        return str(node.get("title", "")).strip()

    if main_num:
        for node in reversed(ancestors):
            cand_title = _title_of(node)
            num_prefix = extract_number_prefix(cand_title)
            if not num_prefix:
                continue
            if not num_prefix.startswith(main_num):
                continue
            if len(num_prefix.split(".")) == 1:
                return cand_title
        if isinstance(main_num_to_anchor_title, dict):
            cand = main_num_to_anchor_title.get(main_num)
            if isinstance(cand, str) and cand.strip():
                return cand.strip()

    if ancestors:
        try:
            oldest_title = _title_of(ancestors[0])
            if oldest_title:
                return oldest_title
        except Exception:
            pass

    best_node = None
    best_level = None
    for node in ancestors:
        try:
            lvl = int(node.get("level") or 0)
        except Exception:
            continue
        if best_level is None or lvl < best_level:
            best_level = lvl
            best_node = node
    if best_node is not None:
        cand_title = _title_of(best_node)
        if cand_title:
            return cand_title

    return title


def select_prompt_path_by_anchor_title(anchor_title: str) -> str:
    """根据锚点标题选择提示词模板。"""
    module_type = classify_module(anchor_title)
    mapping = {
        "title": "mind_map/title_prompts",
        "introduction": "mind_map/introduction_prompts",
        "method": "mind_map/method_prompts",
        "related_work": "mind_map/related_work_prompts",
    }
    return mapping.get(module_type, "mind_map/module_explain_prompts")


def build_module_messages(
    module_title: str,
    module_path: str,
    module_type: str,
    context_text: str,
    anchor_title: str,
) -> Tuple[List[dict], dict]:
    """构造模块解释提示词消息。"""
    system = PromptLoader.load("mind_map/system_prompts")
    user_prompt_path = select_prompt_path_by_anchor_title(anchor_title)
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


def prepare_module_explain_inputs(
    module: dict,
    id_to_module: dict,
    main_num_to_anchor_title: dict,
    request,
) -> dict:
    """整理模块解释的上下文与提示词。"""
    anchor_title = find_chapter_anchor_title(module, main_num_to_anchor_title=main_num_to_anchor_title)
    module_type = classify_module(anchor_title)
    node_ref = module.get("_node_ref") or {}
    subtree_nodes = collect_subtree_nodes(node_ref) if isinstance(node_ref, dict) else []
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

    agg_context_text, agg_context_clipped = truncate_passages(
        aggregated_passages,
        max_chars=request.context_max_chars,
        per_passage_chars=request.context_per_passage_chars,
    )
    subtree_text = build_subtree_text(node_ref, max_chars=request.context_max_chars)
    combined_input = (
        "【模块子树内容】\n"
        f"{subtree_text}\n\n"
        "【子树检索上下文】\n"
        f"{agg_context_text}"
    ).strip()

    messages, prompt_meta = build_module_messages(
        module_title=module.get("title", ""),
        module_path=module.get("path", ""),
        module_type=module_type,
        context_text=combined_input,
        anchor_title=anchor_title,
    )
    return {
        "messages": messages,
        "prompt_meta": prompt_meta,
        "module_type": module_type,
        "anchor_title": anchor_title,
        "subtree_node_count": len(subtree_nodes),
        "context_clipped": agg_context_clipped,
    }


def estimate_tokens_with_fallback(messages: List[dict], answer: str, usage: Optional[dict]) -> Tuple[int, int, int, bool]:
    """解析 token 使用情况，必要时估算。"""
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    estimated = False
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        prompt_tokens = estimate_message_tokens(messages)
        completion_tokens = estimate_token_count(answer)
        total_tokens = prompt_tokens + completion_tokens
        estimated = True
    if not isinstance(total_tokens, int):
        total_tokens = int(prompt_tokens) + int(completion_tokens)
    return int(prompt_tokens), int(completion_tokens), int(total_tokens), estimated
