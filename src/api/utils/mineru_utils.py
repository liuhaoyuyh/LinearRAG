import base64
import json
from pathlib import Path

from src.api.utils.path_utils import safe_filename


def save_mineru_json(result_json: dict, output_dir: Path) -> dict:
    """保存 MinerU 解析结果中的 JSON/图片文件。"""
    output = {}
    results = result_json.get("results", {})
    for file_name, content in results.items():
        safe_name = safe_filename(file_name)
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
                    img_path = images_dir / f"{safe_filename(img_name)}"
                    if img_path.suffix == "":
                        img_path = img_path.with_suffix(suffix)
                    with open(img_path, "wb") as f_img:
                        f_img.write(base64.b64decode(b64_data))
                    output.setdefault("images", []).append(str(img_path))
                except Exception:
                    continue
    return output
