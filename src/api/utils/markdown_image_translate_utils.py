import base64
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont, ImageStat

from prompts.loader import PromptLoader

logger = logging.getLogger(__name__)


MARKDOWN_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HTML_IMAGE_RE = re.compile(r'(?is)<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>')


@dataclass
class OCRBlock:
    text: str
    bbox: Tuple[int, int, int, int]  # 轴对齐边界框 [x1, y1, x2, y2]
    raw_bbox: Optional[Tuple[int, ...]] = None  # 原始坐标（可能是4或5个值）
    is_rotate_rect: bool = False  # 是否来自 rotate_rect 格式


def _rotate_rect_to_bbox_raw(
    cx: float, cy: float, w: float, h: float, angle: float,
) -> Tuple[int, int, int, int]:
    """将 rotate_rect 转换为轴对齐边界框（不做归一化转换）。"""
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    hw, hh = w / 2, h / 2
    corners_rel = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    
    corners_abs = []
    for dx, dy in corners_rel:
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners_abs.append((cx + rx, cy + ry))
    
    xs = [p[0] for p in corners_abs]
    ys = [p[1] for p in corners_abs]
    x1, y1 = int(round(min(xs))), int(round(min(ys)))
    x2, y2 = int(round(max(xs))), int(round(max(ys)))
    
    return x1, y1, x2, y2


def _rotate_rect_to_bbox(
    cx: float, cy: float, w: float, h: float, angle: float,
    img_width: int,
    img_height: int,
    coord_mode: str = "norm1000",
) -> Tuple[int, int, int, int]:
    """将 rotate_rect [cx, cy, w, h, angle] 转换为轴对齐边界框 [x1, y1, x2, y2]。
    
    Args:
        cx, cy: 旋转矩形中心点
        w, h: 宽度和高度（旋转前）
        angle: 逆时针旋转角度（度）
        img_width, img_height: 图像尺寸
        coord_mode: 坐标模式
            - "pixel": 直接像素坐标
            - "norm1000": 0-1000 归一化，w用scale_x，h用scale_y
            - "norm1000_uniform": 0-1000 归一化，w和h用统一缩放（min(scale_x, scale_y)）
            - "norm1000_swap": 0-1000 归一化，w用scale_y，h用scale_x（适合angle=90的情况）
    
    Returns:
        轴对齐边界框 (x1, y1, x2, y2) 像素坐标
    """
    logger.warning(
        "rotate_rect INPUT: cx=%.1f cy=%.1f w=%.1f h=%.1f angle=%.1f img=%dx%d mode=%s",
        cx, cy, w, h, angle, img_width, img_height, coord_mode
    )
    
    scale_x = img_width / 1000.0
    scale_y = img_height / 1000.0
    
    if coord_mode == "pixel":
        # 直接使用像素坐标
        pass
    elif coord_mode == "norm1000":
        # 所有值都是 0-1000 归一化，标准缩放
        cx = cx * scale_x
        cy = cy * scale_y
        w = w * scale_x
        h = h * scale_y
    elif coord_mode == "norm1000_uniform":
        # 所有值都是 0-1000 归一化，w和h用统一缩放
        cx = cx * scale_x
        cy = cy * scale_y
        scale_uniform = min(scale_x, scale_y)
        w = w * scale_uniform
        h = h * scale_uniform
    elif coord_mode == "norm1000_swap":
        # 所有值都是 0-1000 归一化，但 w 用 scale_y，h 用 scale_x
        # 这对于 angle=90 的情况可能更正确（旋转后 w 变成 y 方向，h 变成 x 方向）
        cx = cx * scale_x
        cy = cy * scale_y
        w = w * scale_y  # w 旋转后对应 y 方向
        h = h * scale_x  # h 旋转后对应 x 方向
    
    bbox = _rotate_rect_to_bbox_raw(cx, cy, w, h, angle)
    x1, y1, x2, y2 = bbox
    
    # 限制在图像范围内
    x1 = max(0, min(img_width, x1))
    y1 = max(0, min(img_height, y1))
    x2 = max(0, min(img_width, x2))
    y2 = max(0, min(img_height, y2))
    
    logger.warning(
        "rotate_rect OUTPUT: bbox=[%d,%d,%d,%d] (size: %dx%d)",
        x1, y1, x2, y2, x2-x1, y2-y1
    )
    
    return x1, y1, x2, y2


def _load_ocr_prompts() -> Tuple[str, str]:
    system = PromptLoader.load("markdown/ocr_system_prompts")
    user = PromptLoader.load("markdown/ocr_user_prompts")
    return system.template, user.template


def _load_image_translate_prompts(text: str) -> List[dict]:
    system = PromptLoader.load("markdown/image_translate_system_prompts")
    user = PromptLoader.load("markdown/image_translate_user_prompts")
    return [
        {"role": "system", "content": system.template},
        {"role": "user", "content": user.render(content=text)},
    ]


def _build_ocr_content(image_b64: str, user_text: str) -> List[dict]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": user_text},
    ]


def _open_ocr_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("未设置环境变量 OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def _normalize_bbox(
    raw: List[float],
    img_width: int,
    img_height: int,
    is_normalized: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """将原始 bbox 坐标转换为像素坐标 [x1, y1, x2, y2]。
    
    qwen-vl-ocr 返回的 bbox 是 0-1000 归一化坐标，格式为 [x1, y1, x2, y2]。
    
    Args:
        raw: 原始坐标 [x1, y1, x2, y2]
        img_width, img_height: 图像尺寸
        is_normalized: 是否为 0-1000 归一化坐标（qwen-vl 默认格式）
    
    Returns:
        像素坐标 (x1, y1, x2, y2)
    """
    if len(raw) < 4:
        return None
    try:
        nums = [float(v) for v in raw[:4]]
    except (TypeError, ValueError):
        return None
    
    x1, y1, x2, y2 = nums
    
    # qwen-vl-ocr 返回的是 0-1000 归一化坐标
    if is_normalized:
        scale_x = img_width / 1000.0
        scale_y = img_height / 1000.0
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        logger.info(
            "bbox: norm1000 -> pixels: [%.1f, %.1f, %.1f, %.1f] (img: %dx%d)",
            x1, y1, x2, y2, img_width, img_height
        )
    
    # 转换为整数
    x1, y1, x2, y2 = [int(round(v)) for v in [x1, y1, x2, y2]]
    
    # 限制在图像范围内
    x1 = max(0, min(img_width, x1))
    y1 = max(0, min(img_height, y1))
    x2 = max(0, min(img_width, x2))
    y2 = max(0, min(img_height, y2))
    
    # 检查是否为有效的 xyxy 格式
    if x2 > x1 and y2 > y1:
        return x1, y1, x2, y2
    
    # Fallback: 尝试解释为 xywh 格式（缩放后的值）
    x, y, w, h = x1, y1, x2 - x1, y2 - y1  # 这里已经转换过了，重新计算
    # 重新从原始值计算
    nums = [float(v) for v in raw[:4]]
    if is_normalized:
        scale_x = img_width / 1000.0
        scale_y = img_height / 1000.0
        x = int(round(nums[0] * scale_x))
        y = int(round(nums[1] * scale_y))
        w = int(round(nums[2] * scale_x))
        h = int(round(nums[3] * scale_y))
    else:
        x, y, w, h = [int(round(v)) for v in nums]
    x2 = x + w
    y2 = y + h
    if x2 > x and y2 > y:
        return x, y, x2, y2
    
    return None


def _calc_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """计算两个边界框的 IoU（交并比）。"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def _deduplicate_blocks(blocks: List[OCRBlock], iou_threshold: float = 0.3) -> List[OCRBlock]:
    """去除重复的文本块。
    
    如果两个文本块的 IoU 大于阈值，则认为是重复的，保留第一个。
    """
    if not blocks:
        return blocks
    
    result = []
    for block in blocks:
        is_duplicate = False
        for existing in result:
            iou = _calc_iou(block.bbox, existing.bbox)
            if iou > iou_threshold:
                # 检查文本是否相似（忽略大小写和空格）
                text1 = block.text.lower().replace(" ", "")
                text2 = existing.text.lower().replace(" ", "")
                if text1 == text2 or text1 in text2 or text2 in text1:
                    is_duplicate = True
                    logger.debug(
                        "Duplicate block removed: '%s' (IoU=%.2f with '%s')",
                        block.text, iou, existing.text
                    )
                    break
        if not is_duplicate:
            result.append(block)
    
    if len(result) < len(blocks):
        logger.warning(
            "Removed %d duplicate blocks (from %d to %d)",
            len(blocks) - len(result), len(blocks), len(result)
        )
    
    return result


def _filter_oversized_blocks(
    blocks: List[OCRBlock], 
    img_width: int, 
    img_height: int,
    max_area_ratio: float = 0.15,
    max_dimension_ratio: float = 0.5,
) -> List[OCRBlock]:
    """过滤掉太大的文本块（可能是背景区域被误识别）。
    
    Args:
        blocks: 文本块列表
        img_width, img_height: 图像尺寸
        max_area_ratio: 文本块面积占图像面积的最大比例（默认 15%）
        max_dimension_ratio: 文本块宽度/高度占图像宽度/高度的最大比例（默认 50%）
    """
    if not blocks:
        return blocks
    
    img_area = img_width * img_height
    result = []
    
    for block in blocks:
        x1, y1, x2, y2 = block.bbox
        block_width = x2 - x1
        block_height = y2 - y1
        block_area = block_width * block_height
        
        # 检查面积比例
        area_ratio = block_area / img_area if img_area > 0 else 0
        if area_ratio > max_area_ratio:
            logger.warning(
                "Filtered oversized block (area %.1f%%): '%s'",
                area_ratio * 100, block.text[:30]
            )
            continue
        
        # 检查宽度/高度比例
        width_ratio = block_width / img_width if img_width > 0 else 0
        height_ratio = block_height / img_height if img_height > 0 else 0
        if width_ratio > max_dimension_ratio and height_ratio > max_dimension_ratio:
            logger.warning(
                "Filtered oversized block (size %.1f%% x %.1f%%): '%s'",
                width_ratio * 100, height_ratio * 100, block.text[:30]
            )
            continue
        
        result.append(block)
    
    if len(result) < len(blocks):
        logger.warning(
            "Filtered %d oversized blocks (from %d to %d)",
            len(blocks) - len(result), len(blocks), len(result)
        )
    
    return result


def _parse_ocr_response(
    payload,
    img_width: int,
    img_height: int,
    coord_mode: str = "norm1000_swap",
) -> List[OCRBlock]:
    """解析 OCR 响应，支持 rotate_rect 和 bbox 两种格式。
    
    Args:
        coord_mode: 坐标模式
            - "norm1000_swap": 默认，适合 qwen-vl-ocr（angle=90时w和h缩放方向交换）
            - "norm1000": 标准归一化
            - "pixel": 像素坐标
    """
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("blocks", []) if isinstance(payload, dict) else []
    blocks = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if len(text) == 1:
            continue
        
        # 优先检测 bbox_2d 格式 [x1, y1, x2, y2]
        # 注意：bbox_2d 坐标可能是基于模型内部处理尺寸（约1000x1000），需要缩放到原图尺寸
        bbox_2d = item.get("bbox_2d")
        if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox_2d[:4]]
                raw_bbox = tuple(int(round(v)) for v in bbox_2d[:4])
                
                # 检测是否需要缩放：如果坐标最大值接近 1000，可能是基于 1000x1000 的
                max_coord = max(x1, y1, x2, y2)
                if max_coord > 0 and max_coord <= 1100:
                    # 假设是基于 1000x1000 的坐标，缩放到原图尺寸
                    scale_x = img_width / 1000.0
                    scale_y = img_height / 1000.0
                    x1 = int(round(x1 * scale_x))
                    y1 = int(round(y1 * scale_y))
                    x2 = int(round(x2 * scale_x))
                    y2 = int(round(y2 * scale_y))
                    logger.info(
                        "bbox_2d: raw=%s -> scaled=[%d,%d,%d,%d] (scale: %.2fx%.2f)",
                        raw_bbox, x1, y1, x2, y2, scale_x, scale_y
                    )
                else:
                    # 直接使用像素坐标
                    x1, y1, x2, y2 = [int(round(v)) for v in [x1, y1, x2, y2]]
                    logger.info("bbox_2d: raw=%s -> pixels=[%d,%d,%d,%d]", raw_bbox, x1, y1, x2, y2)
                
                # 边界检查
                x1 = max(0, min(img_width, x1))
                y1 = max(0, min(img_height, y1))
                x2 = max(0, min(img_width, x2))
                y2 = max(0, min(img_height, y2))
                
                if x2 > x1 and y2 > y1:
                    blocks.append(OCRBlock(text=text, bbox=(x1, y1, x2, y2), raw_bbox=raw_bbox))
                continue
            except (TypeError, ValueError) as e:
                logger.warning("Failed to parse bbox_2d: %s", e)

        # 优先检测 rotate_rect 格式 [cx, cy, w, h, angle]
        rotate_rect = item.get("rotate_rect")
        if rotate_rect and isinstance(rotate_rect, list) and len(rotate_rect) >= 5:
            try:
                cx, cy, w, h, angle = [float(v) for v in rotate_rect[:5]]
                raw_bbox = tuple(int(round(float(v))) for v in rotate_rect[:5])
                normalized = _rotate_rect_to_bbox(
                    cx, cy, w, h, angle, 
                    img_width, img_height, 
                    coord_mode=coord_mode
                )
                logger.warning(
                    "rotate_rect: raw=[%.1f,%.1f,%.1f,%.1f,%.1f] -> bbox=%s (mode=%s)",
                    cx, cy, w, h, angle, normalized, coord_mode
                )
                blocks.append(OCRBlock(
                    text=text, 
                    bbox=normalized, 
                    raw_bbox=raw_bbox,
                    is_rotate_rect=True,
                ))
                continue
            except (TypeError, ValueError) as e:
                logger.warning("Failed to parse rotate_rect: %s", e)
        
        # 回退到 bbox 或 box 格式 [x1, y1, x2, y2]
        bbox = item.get("bbox") or item.get("box") or []
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        raw_bbox = None
        try:
            raw_bbox = tuple(int(round(float(v))) for v in bbox[:4])
        except (TypeError, ValueError):
            raw_bbox = None
        # qwen-vl-ocr 返回 0-1000 归一化坐标
        normalized = _normalize_bbox(bbox, img_width, img_height, is_normalized=True)
        if not normalized:
            continue
        logger.info("bbox: raw=%s -> pixels=%s", raw_bbox, normalized)
        blocks.append(OCRBlock(text=text, bbox=normalized, raw_bbox=raw_bbox))
    
    # 去除重复的文本块
    blocks = _deduplicate_blocks(blocks)
    # 过滤太大的文本块
    blocks = _filter_oversized_blocks(blocks, img_width, img_height)
    return blocks


def _parse_ocr_fallback(
    content: str,
    img_width: int,
    img_height: int,
    coord_mode: str = "norm1000_swap",
) -> List[OCRBlock]:
    """从原始文本内容中解析 OCR 结果（JSON 解析失败时的备用方案）。"""
    blocks: List[OCRBlock] = []
    if not content:
        return blocks
    
    # 优先匹配 rotate_rect 格式（5个数字）
    rotate_pattern = re.compile(
        r'"rotate_rect"\s*:\s*\[([^\]]+)\][^{}]*?"text"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    for match in rotate_pattern.finditer(content):
        raw_nums = match.group(1)
        text = match.group(2).strip()
        if not text:
            continue
        if len(text) == 1:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_nums)
        if len(nums) >= 5:
            try:
                cx, cy, w, h, angle = [float(v) for v in nums[:5]]
                raw_bbox = tuple(int(round(float(v))) for v in nums[:5])
                normalized = _rotate_rect_to_bbox(
                    cx, cy, w, h, angle, 
                    img_width, img_height, 
                    coord_mode=coord_mode
                )
                blocks.append(OCRBlock(
                    text=text, 
                    bbox=normalized, 
                    raw_bbox=raw_bbox,
                    is_rotate_rect=True,
                ))
                continue
            except (TypeError, ValueError):
                pass
    
    # 回退匹配 bbox_2d/bbox/box 格式（4个数字）
    bbox_pattern = re.compile(
        r'"(?:bbox_2d|bbox|box)"\s*:\s*\[([^\]]+)\][^{}]*?"text"\s*:\s*"([^"]*)"',
        re.DOTALL,
    )
    for match in bbox_pattern.finditer(content):
        raw_nums = match.group(1)
        text = match.group(2).strip()
        if not text:
            continue
        if len(text) == 1:
            continue
        nums = re.findall(r"-?\d+(?:\.\d+)?", raw_nums)
        if len(nums) < 4:
            continue
        raw_bbox = tuple(int(round(float(v))) for v in nums[:4])
        is_normalized = coord_mode != "pixel"
        normalized = _normalize_bbox(
            [float(v) for v in nums[:4]], 
            img_width, img_height, 
            is_normalized=is_normalized
        )
        if not normalized:
            continue
        blocks.append(OCRBlock(text=text, bbox=normalized, raw_bbox=raw_bbox))
    
    # 去除重复的文本块
    blocks = _deduplicate_blocks(blocks)
    # 过滤太大的文本块
    blocks = _filter_oversized_blocks(blocks, img_width, img_height)
    return blocks


def ocr_image_blocks(image_path: Path, ocr_model: str) -> List[OCRBlock]:
    _, user_template = _load_ocr_prompts()
    buffered = image_path.read_bytes()
    image_b64 = base64.b64encode(buffered).decode("ascii")
    
    # 获取图像尺寸，用于正确解析归一化坐标
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    logger.info("OCR image size: %dx%d", img_width, img_height)

    user_text = user_template
    messages = [
        {"role": "user", "content": _build_ocr_content(image_b64, user_text)},
    ]
    client = _open_ocr_client()
    response = client.chat.completions.create(
        model=ocr_model,
        messages=messages,
        temperature=0.7,
        frequency_penalty=0.8,
        top_p=0.8,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    logger.warning("OCR response content: %s", content)
    try:
        payload = json.loads(content)
        return _parse_ocr_response(payload, img_width, img_height)
    except json.JSONDecodeError as exc:
        logger.warning("OCR response parse failed: %s", exc)
        return _parse_ocr_fallback(content, img_width, img_height)


def _choose_font(size: int) -> ImageFont.ImageFont:
    font_paths = [
        os.getenv("IMAGE_TRANSLATE_FONT_PATH"),
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in font_paths:
        if not path:
            continue
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_text(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    if not words:
        return [text]
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        width = draw.textlength(candidate, font=font)
        if width <= max_width or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) == 1 and draw.textlength(lines[0], font=font) > max_width:
        # No spaces (e.g., Chinese), fall back to char wrapping.
        lines = []
        current = ""
        for ch in text:
            candidate = f"{current}{ch}"
            if draw.textlength(candidate, font=font) <= max_width or not current:
                current = candidate
                continue
            lines.append(current)
            current = ch
        if current:
            lines.append(current)
    return lines


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _sample_background(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """采样文本框周围边缘区域的背景颜色。
    
    只采样边缘像素，避免把文字颜色混入背景色计算。
    使用中位数来获取更准确的背景色。
    """
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    
    # 采样边缘宽度：框尺寸的 15%，最小 3 像素，最大 15 像素
    edge_width = max(3, min(15, int(min(box_width, box_height) * 0.15)))
    
    pixels = []
    
    # 采样四条边缘区域
    regions = []
    
    # 上边缘
    top_y1 = max(0, y1 - edge_width)
    if top_y1 < y1:
        regions.append((x1, top_y1, x2, y1))
    
    # 下边缘
    bottom_y2 = min(image.height, y2 + edge_width)
    if y2 < bottom_y2:
        regions.append((x1, y2, x2, bottom_y2))
    
    # 左边缘
    left_x1 = max(0, x1 - edge_width)
    if left_x1 < x1:
        regions.append((left_x1, y1, x1, y2))
    
    # 右边缘
    right_x2 = min(image.width, x2 + edge_width)
    if x2 < right_x2:
        regions.append((x2, y1, right_x2, y2))
    
    # 从每个边缘区域采样
    for rx1, ry1, rx2, ry2 in regions:
        if rx2 > rx1 and ry2 > ry1:
            region = image.crop((rx1, ry1, rx2, ry2))
            stat = ImageStat.Stat(region)
            # 获取该区域的中位数（使用 median 而不是 mean）
            if hasattr(stat, 'median'):
                pixels.append(tuple(int(v) for v in stat.median[:3]))
            else:
                pixels.append(tuple(int(v) for v in stat.mean[:3]))
    
    if not pixels:
        # 如果没有采样到边缘像素，回退到采样整个区域的平均值
        region = image.crop((x1, y1, x2, y2))
        stat = ImageStat.Stat(region)
        mean = stat.mean[:3]
        return tuple(int(v) for v in mean)
    
    # 计算所有边缘区域颜色的中位数
    r_values = sorted([p[0] for p in pixels])
    g_values = sorted([p[1] for p in pixels])
    b_values = sorted([p[2] for p in pixels])
    
    mid = len(pixels) // 2
    return (r_values[mid], g_values[mid], b_values[mid])


def _contrast_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _protect_terms(text: str) -> Tuple[str, Dict[str, str]]:
    terms: Dict[str, str] = {}

    def _repl(match: re.Match) -> str:
        token = match.group(0)
        key = f"<<TERM_{len(terms) + 1}>>"
        terms[key] = token
        return key

    protected = re.sub(r"\b[A-Z]{2,}[A-Z0-9\-_/]*\b", _repl, text)
    return protected, terms


def _restore_terms(text: str, terms: Dict[str, str]) -> str:
    for key, value in terms.items():
        text = text.replace(key, value)
    return text


def _should_skip_translation(text: str) -> bool:
    """判断文本是否应该跳过翻译（返回 True 表示不需要翻译）。"""
    stripped = text.strip()
    
    # 空文本
    if not stripped:
        return True
    
    # 太短的文本（1-2个字符）
    if len(stripped) <= 2:
        return True
    
    # 纯符号/数字
    if re.fullmatch(r"[^A-Za-z]+", stripped):
        return True
    
    # 不包含英文字母（已经是中文或其他语言）
    if not re.search(r"[A-Za-z]", stripped):
        return True
    
    # 常见的不需要翻译的专有名词/缩写
    skip_patterns = [
        r"^[A-Z]{2,}$",  # 纯大写缩写如 LSTM, FC, CNN, RNN
        r"^[A-Z][a-z]*Net$",  # xxxNet 如 PointNet, ResNet
        r"^[A-Z][a-z]*[A-Z][a-z]*$",  # 驼峰式专有名词
        r"^\([a-z]\)$",  # (a), (b), (c) 等标签
        r"^[A-Z]$",  # 单个大写字母如 R, W
        r"^[A-Z]_?pred$",  # Wpred 等
        r"^[a-z]$",  # 单个小写字母
    ]
    for pattern in skip_patterns:
        if re.fullmatch(pattern, stripped):
            return True
    
    return False


def translate_ocr_text(text: str, llm_model) -> Optional[str]:
    """翻译文本。如果不需要翻译，返回 None。"""
    stripped = text.strip()
    
    if _should_skip_translation(stripped):
        return None  # 返回 None 表示不需要翻译
    
    protected, terms = _protect_terms(stripped)
    messages = _load_image_translate_prompts(protected)
    response = llm_model.generate(messages)
    translated = (getattr(response, "content", "") or "").strip()
    if not translated:
        return None  # 翻译失败，不替换
    translated = _restore_terms(translated, terms)
    
    # 如果翻译结果与原文相同，不需要替换
    if translated.lower().strip() == stripped.lower().strip():
        return None
    
    return translated


def render_translated_blocks(
    image_path: Path,
    blocks: List[OCRBlock],
    llm_model,
    output_path: Path,
) -> None:
    """渲染翻译后的文本到图像上。"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 创建调试图像（使用 norm1000_swap 模式，适合 qwen-vl-ocr）
        debug_img = img.copy()
        debug_draw = ImageDraw.Draw(debug_img)
        
        logger.warning(
            "Render image: %s size=%dx%d blocks=%d",
            str(image_path),
            img.width,
            img.height,
            len(blocks),
        )
        
        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            
            # 在调试图像上绘制边界框
            debug_draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
            
            # 边界检查
            x1 = max(0, min(img.width, x1))
            y1 = max(0, min(img.height, y1))
            x2 = max(0, min(img.width, x2))
            y2 = max(0, min(img.height, y2))
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "Skip invalid block: bbox=%s text=%s",
                    str(block.bbox),
                    block.text[:20],
                )
                continue
            
            translated = translate_ocr_text(block.text, llm_model)
            
            # 如果不需要翻译（返回 None），跳过这个文本块
            if translated is None:
                logger.info(
                    "Skip block (no translation needed): bbox=[%d,%d,%d,%d] text='%s'",
                    x1, y1, x2, y2, block.text.replace("\n", "\\n"),
                )
                continue
            
            logger.info(
                "Render block: bbox=[%d,%d,%d,%d] src='%s' dst='%s'",
                x1, y1, x2, y2,
                block.text.replace("\n", "\\n"),
                translated.replace("\n", "\\n"),
            )
            
            # 采样背景色并填充矩形
            bg = _sample_background(img, (x1, y1, x2, y2))
            draw.rectangle((x1, y1, x2, y2), fill=bg)
            
            # 渲染翻译后的文本
            text_color = _contrast_color(bg)
            box_width = x2 - x1
            box_height = y2 - y1
            font_size = max(10, int(box_height * 0.7))
            font = _choose_font(font_size)
            lines = _wrap_text(translated, draw, font, box_width)
            
            # 自动调整字体大小以适应框
            while lines:
                max_line_width = max(_text_bbox(draw, line, font)[0] for line in lines)
                total_height = sum(_text_bbox(draw, line, font)[1] for line in lines)
                if max_line_width <= box_width and total_height <= box_height:
                    break
                font_size = max(8, font_size - 1)
                if font_size == 8:
                    break
                font = _choose_font(font_size)
                lines = _wrap_text(translated, draw, font, box_width)
            
            # 居中绘制文本
            total_height = sum(_text_bbox(draw, line, font)[1] for line in lines) if lines else 0
            start_y = y1 + max(0, int((box_height - total_height) / 2))
            for line in lines:
                line_width, line_height = _text_bbox(draw, line, font)
                start_x = x1 + max(0, int((box_width - line_width) / 2))
                draw.text((start_x, start_y), line, fill=text_color, font=font)
                start_y += line_height
        
        # 保存翻译后的图像
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        logger.warning("Saved translated image: %s", str(output_path))
        
        # 保存调试图像
        debug_path = output_path.with_name(f"{output_path.stem}_debug{output_path.suffix}")
        debug_img.save(debug_path)
        logger.info("Saved debug image: %s", str(debug_path))


def _extract_md_image_path(raw: str) -> Tuple[str, str]:
    raw = raw.strip()
    if raw.startswith("<") and ">" in raw:
        close_idx = raw.find(">")
        path = raw[1:close_idx]
        rest = raw[close_idx + 1 :]
        return path, rest
    parts = raw.split()
    if not parts:
        return raw, ""
    path = parts[0]
    rest = raw[len(path):]
    return path, rest


def _resolve_image_path(md_dir: Path, image_ref: str) -> Optional[Path]:
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        return None
    candidate = Path(image_ref)
    if not candidate.is_absolute():
        candidate = (md_dir / candidate).resolve()
    return candidate


def translate_markdown_images(
    markdown_text: str,
    md_path: Path,
    llm_model,
    ocr_model: str,
) -> Tuple[str, List[str]]:
    md_dir = md_path.parent
    output_dir = md_dir / "images_translate"
    processed: Dict[Path, Path] = {}
    output_paths: List[str] = []

    def _process_image(image_ref: str) -> Optional[str]:
        image_path = _resolve_image_path(md_dir, image_ref)
        if image_path is None:
            return None
        if not image_path.exists():
            logger.warning("图片不存在，跳过: %s", image_path)
            return None
        if image_path in processed:
            return str(processed[image_path].relative_to(md_dir))
        try:
            blocks = ocr_image_blocks(image_path, ocr_model)
        except Exception as exc:
            logger.warning("OCR 失败，跳过图片 %s: %s", image_path, exc)
            return None
        if not blocks:
            logger.warning("OCR 未返回文本块，跳过图片: %s", image_path)
            return None
        output_name = f"{image_path.stem}_zh{image_path.suffix}"
        output_path = output_dir / output_name
        render_translated_blocks(image_path, blocks, llm_model, output_path)
        processed[image_path] = output_path
        output_paths.append(str(output_path))
        return str(output_path.relative_to(md_dir))

    def _replace_md(match: re.Match) -> str:
        alt = match.group(1)
        raw = match.group(2)
        path, rest = _extract_md_image_path(raw)
        new_rel = _process_image(path)
        if not new_rel:
            return match.group(0)
        return f"![{alt}]({new_rel}{rest})"

    def _replace_html(match: re.Match) -> str:
        src = match.group(1)
        new_rel = _process_image(src)
        if not new_rel:
            return match.group(0)
        return match.group(0).replace(src, new_rel)

    from src.api.utils.markdown_translate_utils import split_code_fences

    blocks = split_code_fences(markdown_text)
    rebuilt: List[str] = []
    for block_type, content in blocks:
        if block_type == "code":
            rebuilt.append(content)
            continue
        content = MARKDOWN_IMAGE_RE.sub(_replace_md, content)
        content = HTML_IMAGE_RE.sub(_replace_html, content)
        rebuilt.append(content)

    return "".join(rebuilt), output_paths
