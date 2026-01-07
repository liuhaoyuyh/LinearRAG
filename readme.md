## 🛠️ **Usage**

### 1️⃣ Install Dependencies

**Step 1: Install Python packages**

此环境仅为linearrag环境，mineru环境需要另装

mineru服务启动命令
```bash
mineru-api --host 127.0.0.1 --port 8888
```
python=3.9

```bash
pip install -r requirements.txt
```

**Step 2: Download Spacy language model**

```bash
python -m spacy download en_core_web_trf
```

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
```

**Step 3: Set up your OpenAI API key**

已经设置好，在.env里

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="your-base-url-here"
```

**Optional: Configure Model Client (multi-backend)**

默认即可

```bash
# Provider selection: openai (default) / mock (offline)
export LLM_PROVIDER="openai"

# Retry & timeout
export LLM_TIMEOUT_S="60"
export LLM_MAX_RETRIES="3"
export LLM_RETRY_BACKOFF_S="0.5"
```


**Step 4: Prepare Embedding Model**

Make sure the embedding model is available at:

```
model/all-mpnet-base-v2/
```

百度网盘链接：

```bash
 https://pan.baidu.com/s/19CMaF0rvysxIIAU2lwrapw?pwd=zmcf
```


### 3️⃣ FastAPI 服务

启动服务（默认 8000 端口）：

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

示例请求：

- 索引构建：

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "medical",
    "embedding_model": "model/all-mpnet-base-v2",
    "spacy_model": "en_core_web_trf",
    "working_dir": "./import"
  }'
```

- 问答（需先完成索引）：

```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "medical",
    "questions": [
      {"question": "Who discovered X?", "answer": "Example answer"}
    ]
  }'
```


#### MinerU 文档解析

```bash
export MINERU_BASE_URL="http://127.0.0.1:8000"   # 可选，默认即此地址
export MINERU_FILE_PARSE_PATH="/file_parse"       # 可选，默认即此路径

curl -X POST http://localhost:8000/mineru/parse \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "data/example.pdf",
    "backend": "pipeline",
    "parse_method": "pipeline",
    "return_md": true,
    "return_images": true
  }'
  
  
  
{
  "file_path": "./data/2021年点云姿态估计.pdf",
  "output_dir": "output",
  "backend": "vlm-mlx-engine",
  "parse_method": "auto",
  "formula_enable": true,
  "table_enable": true,
  "server_url": "127.0.0.1:8888",
  "return_md": true,
  "return_middle_json": true,
  "return_model_output": true,
  "return_content_list": true,
  "return_images": true,
  "response_format_zip": true,
  "start_page_id": 0,
  "end_page_id": 99999
}
```

返回将包含 MinerU 调用状态及落盘路径。默认输出目录 `results/mineru/<文件名>/<timestamp>/`，可通过请求体 `output_dir` 覆盖。
若 MinerU 与 FastAPI 不同端口（FastAPI 默认 8000，MinerU 也常用 8000），需将 `MINERU_BASE_URL` 设置为 MinerU 实际地址或调整 FastAPI 端口，否则可能收到 `detail: Not Found`；若 MinerU 暴露的路径不同，可通过 `MINERU_FILE_PARSE_PATH` 修改。

#### 思维导图生成

根据 MinerU 生成的 Markdown 结构化为树状 JSON（根节点为文件名，节点包含 `id/level/title/content/order`）。
默认仅保留从 **Introduction** 开始到 **Conclusion** 结束（包含两端）的模块范围及其全部子树；若无法定位 Introduction 或 Conclusion，则退化为删除 `Abstract/摘要`、`References/参考文献` 及所有尾部章节（如 `Acknowledgements/致谢`、`Appendix/附录`）及其子树。

```bash
curl -X POST http://localhost:8000/mindmap \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "sustainability-16-02641-v2"
  }'
```

服务会在 `output/mineru/<doc_name>/` 下选取时间戳目录名最大的记录，读取 `<doc_name>/<doc_name>.md`，并返回思维导图树。

#### 思维导图模块解释（并发、非流式）

对思维导图树除根节点外的每个模块：用“模块标题 + 检索到的上下文 + 模块提示词”并发调用大模型，返回每个模块的解释内容（单模块不做流式输出）。
解释范围与 `/mindmap` 一致（Introduction→Conclusion；或退化删除 Abstract/References/尾部章节），并且解释输入为“该模块子树内容 + 子树检索上下文”。
解释完成后会基于同一份 `root` 树（节点含 `llm_answer`）额外生成一份仅包含“标题 + llm_answer”的 Markdown，并落盘到 `results/<dataset_name>/<timestamp>/mindmap_explain.md`，同时在响应中返回 `explain_markdown` 与 `explain_markdown_path`。

前置条件：
- 已生成 `dataset/<doc_name>/chunks.json`（可通过 `/markdown/chunk` 生成）
- 已调用 `/index` 完成索引构建（`dataset_name` 需与 `doc_name` 一致，或在请求中指定 `dataset_name`）

```bash
curl -X POST http://localhost:8000/mindmap/explain \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "sustainability-16-02641-v2",
    "module_max_workers": 8,
    "retrieval_top_k": 5,
    "include_tree": true,
    "include_context": true
  }'
```

批量推理（OpenAI-compatible Batch）示例：

```bash
curl -X POST http://localhost:8000/mindmap/explain \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "sustainability-16-02641-v2",
    "use_batch": true,
    "batch_completion_window": "24h",
    "batch_poll_interval_s": 10,
    "retrieval_top_k": 5,
    "include_tree": true,
    "include_context": true
  }'
```

说明：
- `use_batch` 默认 `false`，需要显式在请求体中开启。
- `batch_completion_window` 默认为 `24h`，可按 OpenAI Batch 要求调整。
- `batch_poll_interval_s` 为轮询间隔秒数，默认 10。

#### content_list 转 chunk

将 MinerU 的 `_content_list.json` 转为标准分块文件 `data/<doc_name>/chunk.json`。

```bash
curl -X POST http://localhost:8000/content/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "sustainability-16-02641-v2"
  }'
```

服务会在 `output/mineru/<doc_name>/` 下选取时间戳目录名最大的目录，读取 `<doc_name>_content_list.json`，按配置的 `chunk_token_size` 与 `chunk_overlap_token_size` 生成分块并写入 `data/<doc_name>/chunk.json`。

#### Markdown 转 chunk

将 MinerU 生成的 Markdown 直接分块写入 `dataset/<doc_name>/chunks.json`。

```bash
curl -X POST http://localhost:8000/markdown/chunk \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "2021年点云姿态估计"
  }'
```

服务会在 `output/mineru/<doc_name>/` 下选取时间戳目录名最大的目录，读取 `<doc_name>/<doc_name>.md`，按配置的 `chunk_token_size` 与 `chunk_overlap_token_size` 空格分词生成分块。

#### Markdown 资产分析（图片/表格/公式）

分析 Markdown 中的图片/表格/公式（表格与公式以图片链接形式提供），结合本地上下文与检索结果生成说明。

```bash
curl -X POST http://localhost:8000/markdown/asset/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "2021年点云姿态估计",
    "asset_markdown": "![](images/xxx.png)"
  }'
```

参数说明（完整字段，可按需覆盖）：
- `doc_name`：文档名，对应 `output/mineru/<doc_name>/`
- `dataset_name`：可选，检索/索引用的数据集名；默认等于 `doc_name`（`dataset/<name>/`）
- `asset_markdown`：Markdown 图片语句（表格/公式以图片链接形式提供）
- `llm_model`：分析/回答的 LLM 模型名
- `embedding_model`：向量模型名或路径（SentenceTransformer）
- `spacy_model`：spaCy 模型名
- `working_dir`：索引输出目录
- `batch_size`：索引/处理批大小
- `max_workers`：并发数（检索/索引等）
- `retrieval_top_k`：检索返回的 top-k 段落数
- `max_iterations`：LinearRAG 迭代次数上限
- `top_k_sentence`：每段 passage 选取的句子数
- `passage_ratio`：段落扩展比率
- `passage_node_weight`：图中 passage 节点权重
- `damping`：迭代阻尼（0-1）
- `iteration_threshold`：迭代停止阈值
- `context_max_chars`：拼接后的上下文最大字符数
- `context_per_passage_chars`：每段 passage 的截断字符数
- `local_context_window_chars`：Markdown 本地上下文窗口字符数

#### Markdown 转 DOCX ✨

将翻译后的 Markdown 文件（`*_translate_with_image.md`）转换为 Microsoft Word DOCX 格式。

**前置要求:**
- 安装 pandoc: `brew install pandoc` (macOS) 或 `sudo apt-get install pandoc` (Linux)
- 已完成 `/markdown/translate` 和 `/markdown/translate_with_image` 接口调用

```bash
curl -X POST http://localhost:8000/mindmap/markdown/to_docx \
  -H "Content-Type: application/json" \
  -d '{
    "doc_name": "2021年点云姿态估计"
  }'
```

**功能特性:**
- ✅ 自动转换行内公式 `$...$` 和段落公式 `$$...$$` 为 Word 公式对象
- ✅ HTML 表格转换为 Word 原生表格
- ✅ 自动提取和引用图片到 `media/` 目录
- ✅ 保留原始换行和段落格式
- ✅ 详细的转换日志

**响应示例:**
```json
{
  "status": "success",
  "doc_name": "2021年点云姿态估计",
  "markdown_path": "/path/to/input_translate_with_image.md",
  "docx_path": "/path/to/input_translate_with_image.docx"
}
```

**输出位置:**  
DOCX 文件保存在与 Markdown 文件相同的目录：  
`output/mineru/<doc_name>/<timestamp>/<doc_name>/<doc_name>_translate_with_image.docx`

**详细文档:**  
- 📖 [完整使用指南](docs/markdown_to_docx_usage.md)
- 🚀 [快速入门](QUICKSTART_MARKDOWN_TO_DOCX.md)
- 🧪 [测试脚本](scripts/test_markdown_to_docx.py)
