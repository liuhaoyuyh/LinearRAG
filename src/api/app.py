from fastapi import FastAPI

from src.api.routers import core, mindmap, mineru


def create_app() -> FastAPI:
    """创建 FastAPI 应用并注册路由。"""
    app = FastAPI(title="LinearRAG FastAPI", version="0.1.0", description="LinearRAG 图检索与问答服务化接口")
    app.include_router(core.router)
    app.include_router(mindmap.router)
    app.include_router(mineru.router)
    return app


app = create_app()
