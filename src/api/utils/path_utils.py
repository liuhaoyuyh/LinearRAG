from datetime import datetime
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from src.api.constants import BASE_DIR


def resolve_path(path_str: str, base: Path = BASE_DIR, allow_missing: bool = False) -> Path:
    """解析相对路径到项目根目录，并在需要时允许不存在。"""
    path_obj = Path(path_str)
    if path_obj.is_absolute():
        return path_obj
    candidate = (base / path_obj).resolve()
    if candidate.exists() or allow_missing:
        return candidate
    return path_obj


def build_output_dir(dataset_name: str) -> str:
    """生成运行输出目录。"""
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_root = Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
    output_dir = (results_root / dataset_name / now).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


def build_mineru_output_dir(request_output_dir: Optional[str], file_name: str) -> Path:
    """生成 MinerU 解析输出目录。"""
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = (
        Path(os.getenv("RESULTS_DIR", BASE_DIR / "results"))
        if request_output_dir is None
        else resolve_path(request_output_dir, allow_missing=True)
    )
    output_dir = (base_dir / "mineru" / file_name / now).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def find_latest_markdown(doc_name: str) -> Path:
    """获取最新 MinerU 解析目录中的 Markdown 路径。"""
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


def find_latest_content_list(doc_name: str) -> Path:
    """获取最新 MinerU 解析目录中的内容列表 JSON。"""
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
        candidates = list(latest_dir.rglob(f"{doc_name}_content_list.json"))
        if candidates:
            content_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}_content_list.json")
    return content_path


def find_latest_markdown_path(doc_name: str) -> Path:
    """获取最新 Markdown 路径（不回退到其他 Markdown 文件）。"""
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


def find_latest_middle_json(doc_name: str) -> Path:
    """获取最新 MinerU 解析目录中的 middle.json 路径。"""
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    middle_path = latest_dir / doc_name / f"{doc_name}_middle.json"
    if not middle_path.exists():
        candidates = list(latest_dir.rglob(f"{doc_name}_middle.json"))
        if candidates:
            middle_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}_middle.json")
    return middle_path


def find_latest_translated_markdown(doc_name: str) -> Path:
    """获取最新翻译 Markdown 路径（_translate.md）。"""
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    md_path = latest_dir / doc_name / f"{doc_name}_translate.md"
    if not md_path.exists():
        candidates = list(latest_dir.rglob(f"{doc_name}_translate.md"))
        if candidates:
            md_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}_translate.md")
    return md_path


def find_latest_translated_with_image_markdown(doc_name: str) -> Path:
    """获取最新带图片翻译的 Markdown 路径（_translate_with_image.md）。"""
    mineru_root = BASE_DIR / "output" / "mineru"
    doc_dir = mineru_root / doc_name
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}")
    timestamp_dirs = sorted([p for p in doc_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not timestamp_dirs:
        raise HTTPException(status_code=404, detail=f"文档 {doc_name} 下未找到时间戳目录")
    latest_dir = timestamp_dirs[-1]
    md_path = latest_dir / doc_name / f"{doc_name}_translate_with_image.md"
    if not md_path.exists():
        candidates = list(latest_dir.rglob(f"{doc_name}_translate_with_image.md"))
        if candidates:
            md_path = candidates[0]
        else:
            # 尝试查找任何 *_translate_with_image.md 文件
            candidates_any = list(latest_dir.rglob("*_translate_with_image.md"))
            if candidates_any:
                md_path = candidates_any[0]
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"在 {latest_dir} 下未找到 {doc_name}_translate_with_image.md"
                )
    return md_path


def safe_filename(name: str) -> str:
    """将文件名收敛到安全的路径片段。"""
    return Path(name).name
