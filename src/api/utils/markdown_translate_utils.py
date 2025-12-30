import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, List, Tuple

from prompts.loader import PromptLoader


@dataclass
class PlaceholderStore:
    prefix: str
    items: List[str]

    def add(self, text: str) -> str:
        token = f"<<{self.prefix}_{len(self.items) + 1}>>"
        self.items.append(text)
        return token

    def replace(self, match: re.Match) -> str:
        return self.add(match.group(0))

    def restore(self, text: str) -> str:
        for idx, original in enumerate(self.items, start=1):
            text = text.replace(f"<<{self.prefix}_{idx}>>", original)
        return text


def split_code_fences(text: str) -> List[Tuple[str, str]]:
    """拆分 Markdown 为代码块与普通文本块。"""
    blocks: List[Tuple[str, str]] = []
    lines = text.splitlines(keepends=True)
    in_code = False
    fence = ""
    buf: List[str] = []

    for line in lines:
        stripped = line.lstrip()
        if not in_code and (stripped.startswith("```") or stripped.startswith("~~~")):
            if buf:
                blocks.append(("text", "".join(buf)))
                buf = []
            in_code = True
            fence = stripped[:3]
            buf.append(line)
            continue

        if in_code:
            buf.append(line)
            if stripped.startswith(fence):
                blocks.append(("code", "".join(buf)))
                buf = []
                in_code = False
                fence = ""
            continue

        buf.append(line)

    if buf:
        blocks.append(("code" if in_code else "text", "".join(buf)))

    return blocks


REFERENCE_HEADING_RE = re.compile(
    r"^(#{1,6})\s*(参考文献|references|refenence)(\b|\s|$|[:：])",
    re.IGNORECASE,
)


def split_references_section(text: str) -> Tuple[str, str]:
    """拆分参考文献段落及其后的内容（仅匹配标题行）。"""
    lines = text.splitlines(keepends=True)
    in_code = False
    fence = ""

    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if not in_code and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_code = True
            fence = stripped[:3]
            continue
        if in_code:
            if stripped.startswith(fence):
                in_code = False
                fence = ""
            continue

        if REFERENCE_HEADING_RE.match(stripped):
            return "".join(lines[:idx]), "".join(lines[idx:])

    return text, ""


def protect_text(text: str) -> Tuple[str, List[PlaceholderStore]]:
    """对公式、行内代码与链接 URL 进行占位保护。"""
    code_store = PlaceholderStore(prefix="CODE", items=[])
    url_store = PlaceholderStore(prefix="URL", items=[])
    latex_store = PlaceholderStore(prefix="LATEX", items=[])

    text = re.sub(r"`[^`]*`", code_store.replace, text)

    def _replace_link(match: re.Match) -> str:
        label = match.group(1)
        url = match.group(2)
        token = url_store.add(url)
        return f"[{label}]({token})"

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace_link, text)

    block_patterns = [
        r"\$\$(.+?)\$\$",
        r"\\\[(.+?)\\\]",
    ]
    for pattern in block_patterns:
        text = re.sub(pattern, latex_store.replace, text, flags=re.DOTALL)

    inline_patterns = [
        r"\\\((.+?)\\\)",
        r"(?<!\\)\$([^\n]+?)(?<!\\)\$",
    ]
    for pattern in inline_patterns:
        text = re.sub(pattern, latex_store.replace, text)

    return text, [code_store, url_store, latex_store]


def restore_text(text: str, stores: List[PlaceholderStore]) -> str:
    """回填占位符为原始内容。"""
    for store in stores:
        text = store.restore(text)
    return text


def split_by_tables(text: str) -> List[Tuple[str, str]]:
    """拆分 HTML 表格与普通文本段。"""
    parts: List[Tuple[str, str]] = []
    last_idx = 0
    for match in re.finditer(r"(?is)<table\b.*?</table>", text):
        if match.start() > last_idx:
            parts.append(("text", text[last_idx:match.start()]))
        parts.append(("table", match.group(0)))
        last_idx = match.end()
    if last_idx < len(text):
        parts.append(("text", text[last_idx:]))
    return parts


def split_long_text(text: str, max_chars: int) -> List[str]:
    """按最大长度切分文本片段。"""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        if end < length:
            slice_text = text[start:end]
            match = re.search(r"\s+\S*$", slice_text)
            if match and match.start() > int(max_chars * 0.6):
                end = start + match.start()
        chunks.append(text[start:end])
        start = end
    return chunks


def translate_html_table(table_html: str, translate_fn: Callable[[str], str]) -> str:
    """翻译 HTML 表格中 th/td 文本内容。"""

    def _replace_cell(match: re.Match) -> str:
        tag = match.group(1)
        attrs = match.group(2) or ""
        content = match.group(3) or ""
        segments = re.split(r"(<[^>]+>)", content)
        translated_segments = []
        for seg in segments:
            if not seg:
                continue
            if seg.startswith("<") and seg.endswith(">"):
                translated_segments.append(seg)
            else:
                translated_segments.append(translate_fn(seg))
        return f"<{tag}{attrs}>{''.join(translated_segments)}</{tag}>"

    return re.sub(r"(?is)<(td|th)([^>]*)>(.*?)</\1>", _replace_cell, table_html)


def build_translation_messages(text: str) -> List[dict]:
    """构造翻译提示词消息。"""
    system = PromptLoader.load("markdown/system_prompts")
    user = PromptLoader.load("markdown/translate_prompts")
    user_text = user.render(content=text)
    return [
        {"role": "system", "content": system.template},
        {"role": "user", "content": user_text},
    ]


def needs_translation(text: str) -> bool:
    """判断文本是否需要翻译。"""
    if not text.strip():
        return False
    scrubbed = re.sub(r"<<[A-Z]+_\\d+>>", "", text)
    return re.search(r"[A-Za-z]", scrubbed) is not None


def translate_text(text: str, llm_model) -> str:
    """调用模型翻译单个文本片段。"""
    if not needs_translation(text):
        return text

    leading = re.match(r"^\s*", text).group(0)
    trailing = re.search(r"\s*$", text).group(0)
    core = text.strip()
    if not core:
        return text

    messages = build_translation_messages(core)
    response = llm_model.generate(messages)
    content = getattr(response, "content", "") or ""
    translated = content.strip() if content.strip() else core
    return f"{leading}{translated}{trailing}"


def translate_text_chunks(chunks: List[str], llm_model, max_workers: int) -> List[str]:
    """并发翻译文本片段。"""
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(translate_text, chunk, llm_model): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return results


def translate_markdown_text(markdown_text: str, llm_model, max_workers: int, chunk_max_chars: int) -> str:
    """翻译 Markdown 文本，保留结构并保护代码/公式。"""
    translatable_text, tail_text = split_references_section(markdown_text)
    blocks = split_code_fences(translatable_text)
    translated_blocks: List[str] = []

    for kind, block_text in blocks:
        if kind == "code":
            translated_blocks.append(block_text)
            continue

        protected_text, stores = protect_text(block_text)
        table_parts = split_by_tables(protected_text)
        translated_parts: List[str] = []

        for part_kind, part_text in table_parts:
            if part_kind == "table":
                translated_parts.append(translate_html_table(part_text, lambda t: translate_text(t, llm_model)))
                continue

            segments = re.split(r"(\n\s*\n)", part_text)
            chunk_indices: List[int] = []
            chunks: List[str] = []
            rebuilt: List[str] = []

            for seg in segments:
                if not seg:
                    continue
                if re.fullmatch(r"\n\s*\n", seg):
                    rebuilt.append(seg)
                    continue
                split_chunks = split_long_text(seg, chunk_max_chars)
                for chunk in split_chunks:
                    chunk_indices.append(len(rebuilt))
                    rebuilt.append(chunk)
                    chunks.append(chunk)

            if chunks:
                translated_chunks = translate_text_chunks(chunks, llm_model, max_workers=max_workers)
                for idx, translated in zip(chunk_indices, translated_chunks):
                    rebuilt[idx] = translated

            translated_parts.append("".join(rebuilt))

        translated_block = "".join(translated_parts)
        translated_blocks.append(restore_text(translated_block, stores))

    return "".join(translated_blocks) + tail_text
