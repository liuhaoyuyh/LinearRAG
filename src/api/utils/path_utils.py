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
    
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    md_path = latest_dir / doc_name / f"{doc_name}.md"
    if md_path.exists():
        return md_path
    
    # 2. 尝试下划线版本
    md_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}.md"
    if md_path.exists():
        return md_path
    
    # 3. 使用 rglob 查找原始文件名
    same_name = list(latest_dir.rglob(f"{doc_name}.md"))
    if same_name:
        return same_name[0]
    
    # 4. 使用 rglob 查找下划线版本
    same_name = list(latest_dir.rglob(f"{doc_name_underscore}.md"))
    if same_name:
        return same_name[0]
    
    # 5. 查找任何 .md 文件作为最后的回退
    candidates = list(latest_dir.rglob("*.md"))
    if candidates:
        return candidates[0]
    
    raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 Markdown 文件")


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
    
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    content_path = latest_dir / doc_name / f"{doc_name}_content_list.json"
    if content_path.exists():
        return content_path
    
    # 2. 尝试下划线版本
    content_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}_content_list.json"
    if content_path.exists():
        return content_path
    
    # 3. 使用 rglob 查找原始文件名
    candidates = list(latest_dir.rglob(f"{doc_name}_content_list.json"))
    if candidates:
        return candidates[0]
    
    # 4. 使用 rglob 查找下划线版本
    candidates = list(latest_dir.rglob(f"{doc_name_underscore}_content_list.json"))
    if candidates:
        return candidates[0]
    
    # 5. 查找任何 *_content_list.json 文件作为最后的回退
    candidates = list(latest_dir.rglob("*_content_list.json"))
    if candidates:
        return candidates[0]
    
    raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 content_list.json 文件（尝试了 {doc_name} 和 {doc_name_underscore}）")


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
    
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    md_path = latest_dir / doc_name / f"{doc_name}.md"
    if md_path.exists():
        return md_path
    
    # 2. 尝试下划线版本
    md_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}.md"
    if md_path.exists():
        return md_path
    
    # 3. 使用 rglob 查找原始文件名
    candidates = list(latest_dir.rglob(f"{doc_name}.md"))
    if candidates:
        return candidates[0]
    
    # 4. 使用 rglob 查找下划线版本
    candidates = list(latest_dir.rglob(f"{doc_name_underscore}.md"))
    if candidates:
        return candidates[0]
    
    raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 {doc_name}.md 或 {doc_name_underscore}.md")


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
    
    # 尝试多种文件名变体
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    middle_path = latest_dir / doc_name / f"{doc_name}_middle.json"
    if middle_path.exists():
        return middle_path
    
    # 2. 尝试下划线版本的 doc_name
    middle_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}_middle.json"
    if middle_path.exists():
        return middle_path
    
    # 3. 使用 rglob 查找原始文件名
    candidates = list(latest_dir.rglob(f"{doc_name}_middle.json"))
    if candidates:
        return candidates[0]
    
    # 4. 使用 rglob 查找下划线版本
    candidates = list(latest_dir.rglob(f"{doc_name_underscore}_middle.json"))
    if candidates:
        return candidates[0]
    
    # 5. 查找任何 *_middle.json 文件作为最后的回退
    candidates = list(latest_dir.rglob("*_middle.json"))
    if candidates:
        return candidates[0]
    
    raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 middle.json 文件（尝试了 {doc_name} 和 {doc_name_underscore}）")


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
    
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    md_path = latest_dir / doc_name / f"{doc_name}_translate.md"
    if md_path.exists():
        return md_path
    
    # 2. 尝试下划线版本
    md_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}_translate.md"
    if md_path.exists():
        return md_path
    
    # 3. 使用 rglob 查找原始文件名
    candidates = list(latest_dir.rglob(f"{doc_name}_translate.md"))
    if candidates:
        return candidates[0]
    
    # 4. 使用 rglob 查找下划线版本
    candidates = list(latest_dir.rglob(f"{doc_name_underscore}_translate.md"))
    if candidates:
        return candidates[0]
    
    # 5. 查找任何 *_translate.md 文件作为最后的回退
    candidates = list(latest_dir.rglob("*_translate.md"))
    if candidates:
        return candidates[0]
    
    raise HTTPException(status_code=404, detail=f"在 {latest_dir} 下未找到 _translate.md 文件（尝试了 {doc_name} 和 {doc_name_underscore}）")


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
    
    doc_name_underscore = doc_name.replace(" ", "_")
    
    # 1. 尝试原始 doc_name
    md_path = latest_dir / doc_name / f"{doc_name}_translate_with_image.md"
    if md_path.exists():
        return md_path
    
    # 2. 尝试下划线版本
    md_path = latest_dir / doc_name_underscore / f"{doc_name_underscore}_translate_with_image.md"
    if md_path.exists():
        return md_path
    
    # 3. 使用 rglob 查找原始文件名
    candidates = list(latest_dir.rglob(f"{doc_name}_translate_with_image.md"))
    if candidates:
        return candidates[0]
    
    # 4. 使用 rglob 查找下划线版本
    candidates = list(latest_dir.rglob(f"{doc_name_underscore}_translate_with_image.md"))
    if candidates:
        return candidates[0]
    
    # 5. 查找任何 *_translate_with_image.md 文件作为最后的回退
    candidates_any = list(latest_dir.rglob("*_translate_with_image.md"))
    if candidates_any:
        return candidates_any[0]
    
    raise HTTPException(
        status_code=404, 
        detail=f"在 {latest_dir} 下未找到 _translate_with_image.md 文件（尝试了 {doc_name} 和 {doc_name_underscore}）"
    )


def safe_filename(name: str) -> str:
    """将文件名收敛到安全的路径片段。"""
    return Path(name).name


def find_doc_directory(doc_name: str) -> Path:
    """
    查找文档目录，支持文件名中的空格变换。
    
    Args:
        doc_name: 文档名称
        
    Returns:
        找到的文档目录路径
        
    Raises:
        HTTPException: 如果找不到文档目录
    """
    mineru_root = BASE_DIR / "output" / "mineru"
    
    # 1. 尝试原始 doc_name
    doc_dir = mineru_root / doc_name
    if doc_dir.exists():
        return doc_dir
    
    # 2. 尝试下划线版本
    doc_name_underscore = doc_name.replace(" ", "_")
    doc_dir = mineru_root / doc_name_underscore
    if doc_dir.exists():
        return doc_dir
    
    # 3. 查找所有匹配的目录（不区分空格/下划线）
    doc_name_normalized = doc_name.replace(" ", "").replace("_", "")
    for potential_dir in mineru_root.iterdir():
        if potential_dir.is_dir():
            potential_name_normalized = potential_dir.name.replace(" ", "").replace("_", "")
            if potential_name_normalized == doc_name_normalized:
                return potential_dir
    
    raise HTTPException(
        status_code=404, 
        detail=f"未在 {mineru_root} 下找到文档目录: {doc_name}"
    )
