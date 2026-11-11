#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · Stage 1 · Web 后端（FastAPI）

把 LangGraph 的 RAG 流水线包装成 HTTP 接口，供前端调用。

接口：
  POST /api/ingest   建索引（传入文档目录）
  POST /api/chat     问答（传入问题）
  GET  /             返回前端页面

启动：uvicorn app:app --reload --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qa import build_qa_pipeline

app = FastAPI(title="local-rag-qa", description="本地知识库问答 · LangGraph 内核")

# 全局持有编译好的图，避免每次请求重建索引
_GRAPH = None
_DOCS_DIR = os.environ.get("RAG_DOCS_DIR", "./sample_docs")


class IngestRequest(BaseModel):
    docs_dir: str


class ChatRequest(BaseModel):
    question: str


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """重建索引。传入本地文档目录路径。"""
    global _GRAPH, _DOCS_DIR
    _DOCS_DIR = req.docs_dir
    graph, vectorstore = build_qa_pipeline(req.docs_dir)
    _GRAPH = graph
    # Qdrant 的向量块数量，从 collection info 里取
    client = vectorstore.client
    info = client.get_collection(vectorstore.collection_name)
    return {"ok": True, "docs_dir": req.docs_dir, "chunks": info.points_count}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """问答。返回答案 + 检索到的证据片段。"""
    global _GRAPH
    if _GRAPH is None:
        # 未显式建索引时，用默认目录建
        if not Path(_DOCS_DIR).exists():
            return {"error": "尚未建索引。请先 POST /api/ingest 传入文档目录。"}
        _GRAPH, _ = build_qa_pipeline(_DOCS_DIR)

    result = _GRAPH.invoke(
        {"question": req.question, "documents": [], "sources": [], "answer": ""}
    )
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "documents": result["documents"],
    }


# 静态前端
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))
