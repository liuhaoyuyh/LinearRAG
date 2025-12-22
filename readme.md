```bash
uvicorn src.api_server:app --host localhost --port 8000
```

### 下载spacy语言模型
```bash
python -m spacy download en_core_web_trf
```