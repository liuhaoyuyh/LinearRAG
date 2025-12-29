import mimetypes
import os
import zipfile

import httpx
from fastapi import APIRouter, HTTPException

from src.api.schemas import MineruParseRequest
from src.api.utils.mineru_utils import save_mineru_json
from src.api.utils.path_utils import build_mineru_output_dir, resolve_path

router = APIRouter()


@router.post("/mineru/parse")
def run_mineru_parse(request: MineruParseRequest):
    """调用 MinerU 服务解析文件。"""
    try:
        file_path = resolve_path(request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=400, detail=f"文件不存在: {file_path}")

        mineru_path = os.getenv("MINERU_FILE_PARSE_PATH", "/file_parse")
        mineru_path = f"/{mineru_path.lstrip('/')}"
        mineru_base = request.server_url
        if not mineru_base.startswith("http://"):
            mineru_base = f"http://{mineru_base}"
        mineru_url = f"{mineru_base}{mineru_path}"

        output_dir = build_mineru_output_dir(request.output_dir, file_path.stem)

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

        saved = save_mineru_json(result_json, output_dir)
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
