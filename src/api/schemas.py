from typing import List, Optional

from pydantic import BaseModel, Field


class BaseRunConfig(BaseModel):
    dataset_name: str = Field(..., description="数据集名称，需存在于 dataset/<name>/")
    embedding_model: str = Field("model/all-mpnet-base-v2", description="SentenceTransformer 模型名称或路径")
    spacy_model: str = Field("en_core_web_trf", description="spaCy 模型名称")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    working_dir: str = Field("./import", description="索引输出目录")
    batch_size: int = Field(128, ge=1)
    max_workers: int = Field(16, ge=1)
    retrieval_top_k: int = Field(5, ge=1)
    max_iterations: int = Field(3, ge=1)
    top_k_sentence: int = Field(1, ge=1)
    passage_ratio: float = Field(1.5, gt=0)
    passage_node_weight: float = Field(0.05, gt=0)
    damping: float = Field(0.5, gt=0, lt=1)
    iteration_threshold: float = Field(0.5, gt=0)


class IndexRequest(BaseRunConfig):
    """索引请求参数"""


class QuestionItem(BaseModel):
    question: str
    answer: Optional[str] = Field(default=None, description="可选：标注答案，用于后续评测")


class QARequest(BaseRunConfig):
    questions: List[QuestionItem]


class EvaluateRequest(BaseModel):
    dataset_name: str = Field(..., description="数据集名称，用于返回信息")
    predictions_path: str = Field(..., description="预测结果 JSON 路径")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    max_workers: int = Field(16, ge=1)


class MineruParseRequest(BaseModel):
    file_path: str = Field(..., description="本地文件路径（PDF/MD/图片等）")
    output_dir: Optional[str] = Field(None, description="解析结果输出目录，默认 results/mineru/<timestamp>/")
    backend: str = Field("pipeline", description="MinerU 后端类型")
    parse_method: str = Field("pipeline", description="PDF 解析方法")
    formula_enable: bool = Field(True, description="是否启用公式解析")
    table_enable: bool = Field(True, description="是否启用表格解析")
    server_url: Optional[str] = Field(None, description="VLM server URL，仅部分后端使用")
    return_md: bool = Field(True, description="是否返回 Markdown 内容")
    return_middle_json: bool = Field(False, description="是否返回中间 JSON")
    return_model_output: bool = Field(False, description="是否返回模型输出 JSON")
    return_content_list: bool = Field(False, description="是否返回内容列表 JSON")
    return_images: bool = Field(False, description="是否返回提取的图片")
    response_format_zip: bool = Field(False, description="是否以 ZIP 格式返回结果")
    start_page_id: int = Field(0, ge=0, description="PDF 解析起始页（从 0 开始）")
    end_page_id: int = Field(99999, ge=0, description="PDF 解析结束页（从 0 开始）")


class MindmapRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")


class ContentChunkRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")


class MarkdownChunkRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")


class MarkdownTranslateRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    max_workers: int = Field(8, ge=1, description="翻译并发数")
    chunk_max_chars: int = Field(2000, ge=200, description="单次翻译最大字符数")


class MarkdownTranslateResponse(BaseModel):
    status: str
    doc_name: str
    markdown_path: str
    translated_path: str
    middle_translate_path: str


class MarkdownTranslateWithImageRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名，读取 <name>_translate.md）")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    ocr_model: str = Field("qwen-vl-ocr", description="OCR 模型名称")


class MarkdownTranslateWithImageResponse(BaseModel):
    status: str
    doc_name: str
    markdown_path: str
    translated_path: str
    image_count: int


class MarkdownAssetAnalyzeRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")
    dataset_name: Optional[str] = Field(
        default=None,
        description="可选：用于检索/索引的数据集名；默认等于 doc_name（对应 dataset/<name>/）",
    )
    asset_markdown: str = Field(..., description="Markdown 图片语句（表格/公式以图片链接形式提供）")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    embedding_model: str = Field("model/all-mpnet-base-v2", description="SentenceTransformer 模型名称或路径")
    spacy_model: str = Field("en_core_web_trf", description="spaCy 模型名称")
    working_dir: str = Field("./import", description="索引输出目录")
    batch_size: int = Field(128, ge=1)
    max_workers: int = Field(16, ge=1)
    retrieval_top_k: int = Field(5, ge=1, description="检索返回的 top-k 上下文数量")
    max_iterations: int = Field(3, ge=1)
    top_k_sentence: int = Field(1, ge=1)
    passage_ratio: float = Field(1.5, gt=0)
    passage_node_weight: float = Field(0.05, gt=0)
    damping: float = Field(0.5, gt=0, lt=1)
    iteration_threshold: float = Field(0.5, gt=0)
    context_max_chars: int = Field(8000, ge=1, description="检索上下文最大字符数（拼接后）")
    context_per_passage_chars: int = Field(1500, ge=1, description="每段 passage 截断的最大字符数")
    local_context_window_chars: int = Field(1200, ge=0, description="Markdown 本地上下文窗口字符数")


class MarkdownAssetAnalyzeResponse(BaseModel):
    analysis: str


class MindmapExplainRequest(BaseModel):
    doc_name: str = Field(..., description="文档名（对应 output/mineru/<name>/ 下目录名）")
    dataset_name: Optional[str] = Field(
        default=None,
        description="可选：用于检索/索引的数据集名；默认等于 doc_name（对应 dataset/<name>/）",
    )
    embedding_model: str = Field("model/all-mpnet-base-v2", description="SentenceTransformer 模型名称或路径")
    spacy_model: str = Field("en_core_web_trf", description="spaCy 模型名称")
    llm_model: str = Field("qwen3-vl-flash", description="OpenAI ChatCompletions 模型名称")
    working_dir: str = Field("./import", description="索引输出目录")
    batch_size: int = Field(128, ge=1)
    retrieval_top_k: int = Field(5, ge=1, description="每个模块检索返回的 top-k 上下文数量")
    max_workers: int = Field(16, ge=1, description="检索/索引并发参数（沿用 LinearRAG 配置）")
    max_iterations: int = Field(3, ge=1)
    top_k_sentence: int = Field(1, ge=1)
    passage_ratio: float = Field(1.5, gt=0)
    passage_node_weight: float = Field(0.05, gt=0)
    damping: float = Field(0.5, gt=0, lt=1)
    iteration_threshold: float = Field(0.5, gt=0)

    module_max_workers: int = Field(8, ge=1, description="模块解释并发数（LLM 调用并发）")
    use_batch: bool = Field(
        default=False,
        description="是否启用 batch 推理（默认关闭）",
    )
    batch_completion_window: str = Field(
        default="24h",
        description="batch completion window（如 24h）",
    )
    batch_poll_interval_s: int = Field(
        default=10,
        description="batch 轮询间隔秒数",
    )
    include_tree: bool = Field(True, description="是否返回带回答的树结构")
    include_context: bool = Field(True, description="是否在结果中包含检索到的上下文")
    include_breadcrumb_in_query: bool = Field(
        False,
        description="是否在检索 query 中拼接节点路径（更强语境，可能更慢/更长）",
    )
    context_max_chars: int = Field(8000, ge=1, description="每个模块上下文最大字符数（拼接后）")
    context_per_passage_chars: int = Field(1500, ge=1, description="每段 passage 截断的最大字符数")


class MindmapExplainResultItem(BaseModel):
    id: str
    title: str
    level: int
    path: str
    module_type: Optional[str] = None
    anchor_title: Optional[str] = None
    subtree_node_count: Optional[int] = None
    prompt_id: Optional[str] = None
    prompt_version: Optional[str] = None
    context: Optional[List[str]] = None
    llm_answer: Optional[str] = None
    error: Optional[str] = None


class MindmapExplainResponse(BaseModel):
    status: str
    doc_name: str
    dataset_name: str
    markdown_path: str
    explain_markdown_path: Optional[str] = None
    explain_markdown: Optional[str] = None
    module_count: int
    results: List[MindmapExplainResultItem]
    root: Optional[dict] = None
    log_path: Optional[str] = None
