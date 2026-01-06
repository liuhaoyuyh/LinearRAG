import base64
import logging
import mimetypes
import re
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException

from prompts.loader import PromptLoader
from src.api.utils.markdown_image_translate_utils import _resolve_image_path

logger = logging.getLogger(__name__)

MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def parse_markdown_image(asset_markdown: str) -> Tuple[str, str, str]:
    """解析 Markdown 图片语句，返回 (alt_text, image_ref, title_text)。"""
    match = MARKDOWN_IMAGE_RE.search(asset_markdown or "")
    if not match:
        raise HTTPException(status_code=400, detail="无法解析 Markdown 图片语句")
    alt_text = (match.group(1) or "").strip()
    raw_ref = (match.group(2) or "").strip()
    image_ref, title_text = _split_image_ref(raw_ref)
    return alt_text, image_ref, title_text


def _split_image_ref(raw_ref: str) -> Tuple[str, str]:
    raw_ref = raw_ref.strip()
    if raw_ref.startswith("<") and ">" in raw_ref:
        close_idx = raw_ref.find(">")
        path = raw_ref[1:close_idx].strip()
        rest = raw_ref[close_idx + 1 :].strip()
        return path, _strip_quotes(rest)
    parts = raw_ref.split()
    if not parts:
        return raw_ref, ""
    path = parts[0].strip()
    rest = raw_ref[len(path) :].strip()
    return path, _strip_quotes(rest)


def _strip_quotes(text: str) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1].strip()
    return stripped


def build_asset_query(alt_text: str, title_text: str, image_ref: str) -> str:
    if alt_text:
        return alt_text
    if title_text:
        return title_text
    if image_ref:
        return Path(image_ref).stem
    return "image"


def extract_local_context(md_text: str, asset_markdown: str, image_ref: str, window_chars: int) -> str:
    if not md_text:
        return ""
    idx = md_text.find(asset_markdown)
    if idx < 0 and image_ref:
        idx = md_text.find(image_ref)
    if idx < 0:
        return ""
    start = max(0, idx - max(0, window_chars))
    end = min(len(md_text), idx + len(asset_markdown) + max(0, window_chars))
    return md_text[start:end].strip()


def resolve_asset_image(md_path: Path, image_ref: str) -> Tuple[Optional[Path], Optional[str]]:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return None, image_ref
    image_path = _resolve_image_path(md_path.parent, image_ref)
    if image_path is None:
        return None, None
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"图片不存在: {image_path}")
    return image_path, None


def load_image_data_url(image_path: Path) -> str:
    data = image_path.read_bytes()
    image_b64 = base64.b64encode(data).decode("ascii")
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{image_b64}"


def build_asset_messages(
    asset_markdown: str,
    query: str,
    local_context: str,
    retrieval_context: str,
    image_url: Optional[str],
) -> Tuple[list, dict]:
    system = PromptLoader.load("markdown/asset_analyze_system_prompts")
    user = PromptLoader.load("markdown/asset_analyze_user_prompts")
    user_text = user.render(
        query=query,
        asset_markdown=asset_markdown,
        local_context=local_context,
        retrieval_context=retrieval_context,
    )
    if image_url:
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        content = user_text
    messages = [
        {"role": "system", "content": system.template},
        {"role": "user", "content": content},
    ]
    meta = {"prompt_id": user.id, "prompt_version": user.version}
    logger.info("Loaded prompt %s version %s", user.id, user.version)
    return messages, meta
