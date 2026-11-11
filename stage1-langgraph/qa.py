#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · Stage 1 · LangGraph 内核

同一个「本地知识库问答」，用不同 Agent 框架各实现一遍。
这是第一阶段：用 LangGraph 的 StateGraph 串起 RAG 流水线。

本文件是「核心引擎」，同时服务两种入口：
1. CLI：python qa.py <文档目录> [问题]
2. Web：app.py 里的 FastAPI 接口调用 build_qa_pipeline()

内核要暴露的点（LangGraph 的设计）：
1. StateGraph —— 把「问答」建模成一张有状态的图，节点是步骤，边是流转。
2. TypedDict 状态 —— 整张图共享一份带类型的状态，节点只改自己关心的字段。
3. 显式节点 / 边 —— retrieve 和 generate 是两个独立节点，数据靠状态传递。
4. 可扩展 —— 之后想加「重写问题」「多路召回」「重排序」，都是往图里加节点。
"""

from typing import TypedDict, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 1. 状态：整张图共享的唯一数据载体
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    question: str          # 用户提问
    documents: List[str]   # 检索到的文档片段（内容）
    sources: List[str]     # 检索到的片段来源（用于前端展示「证据」）
    answer: str            # 最终答案


# ---------------------------------------------------------------------------
# 2. 节点一：检索（retrieve）
#    从向量库里召回和问题最相关的片段
# ---------------------------------------------------------------------------
def _make_retrieve(retriever):
    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {
            "documents": [d.page_content for d in docs],
            "sources": [d.metadata.get("source", "未知来源") for d in docs],
        }
    return retrieve


# ---------------------------------------------------------------------------
# 3. 节点二：生成（generate）
#    把召回的片段拼进 prompt，交给 LLM 生成带依据的答案
# ---------------------------------------------------------------------------
def _make_generate(llm):
    def generate(state: RAGState) -> dict:
        context = "\n\n".join(
            f"[{i+1}] {doc}" for i, doc in enumerate(state["documents"])
        )
        prompt = (
            "你是一个严谨的知识库问答助手。\n"
            "规则：只根据下面给定的上下文回答，不要编造上下文之外的内容。\n"
            "如果上下文不足以回答，请直接说「根据现有资料无法回答」。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{state['question']}"
        )
        answer = llm.invoke(prompt).content
        return {"answer": answer}
    return generate


# ---------------------------------------------------------------------------
# 4. 建图：节点 + 边 + 编译
# ---------------------------------------------------------------------------
def build_graph(retriever, llm) -> object:
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", _make_retrieve(retriever))
    graph.add_node("generate", _make_generate(llm))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. 索引构建：把本地文档切块、向量化、存入 Chroma
# ---------------------------------------------------------------------------
def build_index(docs_dir: str, persist_dir: str = "./chroma_db"):
    loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    return vectorstore


# ---------------------------------------------------------------------------
# 6. 组装完整流水线：返回 (compiled_graph, vectorstore)
# ---------------------------------------------------------------------------
def build_qa_pipeline(docs_dir: str, persist_dir: str = "./chroma_db", model: str = "qwen3:8b"):
    """给 Web 后端用：建索引 + 组装图，返回 (graph, vectorstore)。"""
    vectorstore = build_index(docs_dir, persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model=model, temperature=0)
    graph = build_graph(retriever, llm)
    return graph, vectorstore


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main():
    import sys

    if len(sys.argv) < 2:
        print("用法：python qa.py <文档目录> [问题]")
        print("  第一次运行会先建索引，之后直接问答。")
        return

    docs_dir = sys.argv[1]

    print("正在构建索引...")
    graph, _ = build_qa_pipeline(docs_dir)

    question = sys.argv[2] if len(sys.argv) >= 3 else input("请输入你的问题：")

    result = graph.invoke(
        {"question": question, "documents": [], "sources": [], "answer": ""}
    )
    print("\n" + "=" * 40)
    print("问题：", question)
    print("-" * 40)
    print("答案：", result["answer"])


if __name__ == "__main__":
    main()
