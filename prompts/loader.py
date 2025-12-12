import os
import yaml
from dataclasses import dataclass
from jinja2 import Template
from pathlib import Path
from functools import lru_cache

PROMPT_ROOT = Path(__file__).resolve().parent


@dataclass
class Prompt:
    """
    Structured representation of a prompt, including metadata and template.
    """

    id: str
    version: str
    description: str
    template: str

    def render(self, **kwargs) -> str:
        # 优先支持 Jinja 语法（包含 "{{" 或 "{%"），否则回退到 str.format() 以兼容 {var}
        if "{{" in self.template or "{%" in self.template:
            template = Template(self.template)
            return template.render(**kwargs)
        return self.template.format(**kwargs)


class PromptLoader:

    @staticmethod
    @lru_cache(maxsize=128)
    def load(path: str) -> Prompt:
        """
        加载提示词 YAML 文件
        例如 path="video/scene_plan" → src/prompts/video/scene_plan.yaml
        """
        file_path = PROMPT_ROOT / f"{path}.yaml"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        template_str = data.get("template", "")
        # 如果缺失，使用路径推导出 id，保持向后兼容
        prompt_id = data.get("id") or path.replace("/", ".")
        version = str(data.get("version", ""))
        description = data.get("description", "")

        return Prompt(
            id=prompt_id,
            version=version,
            description=description,
            template=template_str,
        )

    @staticmethod
    def render(path: str, **kwargs) -> str:
        """
        渲染提示词模板
        """
        prompt = PromptLoader.load(path)
        return prompt.render(**kwargs)

    @staticmethod
    def metadata(path: str) -> dict:
        """
        获取提示词元数据（id, version, description）
        """
        prompt = PromptLoader.load(path)
        return {
            "id": prompt.id,
            "version": prompt.version,
            "description": prompt.description,
        }
