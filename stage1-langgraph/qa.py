#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · Stage 1 · LangGraph 内核

同一个「本地知识库问答」，用不同 Agent 框架各实现一遍。
这是第一阶段：用 LangGraph 的 StateGraph 串起 RAG 流水线。

内核要暴露的点（LangGraph 的设计）：
1. StateGraph —— 把「问答」建模成一张有状态的图，节点是步骤，边是流转。
2. TypedDict 状态 —— 整张图共享一份带类型的状态，节点只改自己关心的字段。
3. 显式节点 / 边 —— retrieve 和 generate 是两个独立节点，数据靠状态传递。
4. 可扩展 —— 之后想加「重写问题」「多路召回」「重排序」，都是往图里加节点。

依赖：pip install langgraph langchain-ollama langchain-community chromadb
模型：本地 Ollama（ollama pull qwen3:8b 或 llama3.2）
"""

from typing import TypedDict, List

from langchain_community.document_loaders import TextLoader, DirectoryLoader
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
    documents: List[str]   # 检索到的文档片段
    answer: str            # 最终答案


# ---------------------------------------------------------------------------
# 2. 节点一：检索（retrieve）
#    从向量库里召回和问题最相关的片段
# ---------------------------------------------------------------------------
def retrieve(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": [d.page_content for d in docs]}


# ---------------------------------------------------------------------------
# 3. 节点二：生成（generate）
#    把召回的片段拼进 prompt，交给 LLM 生成带依据的答案
# ---------------------------------------------------------------------------
def generate(state: RAGState) -> dict:
    context = "\n\n".join(state["documents"])
    prompt = (
        "只根据下面给定的上下文回答，不要编造上下文之外的内容。\n"
        f"上下文：\n{context}\n\n"
        f"问题：{state['question']}"
    )
    answer = llm.invoke(prompt).content
    return {"answer": answer}


# ---------------------------------------------------------------------------
# 4. 建图：节点 + 边 + 编译
#    retrieve -> generate -> END，一条直线，但每个节点都是可替换的独立单元
# ---------------------------------------------------------------------------
def build_graph(retriever_obj, llm_obj):
    global retriever, llm
    retriever = retriever_obj
    llm = llm_obj

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
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


def main():
    import sys

    if len(sys.argv) < 2:
        print("用法：python qa.py <文档目录> [问题]")
        print("  第一次运行会先建索引，之后直接问答。")
        return

    docs_dir = sys.argv[1]

    # 索引（首次较慢，之后复用 persist_dir）
    print("正在构建索引...")
    vectorstore = build_index(docs_dir)
    retriever_obj = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm_obj = ChatOllama(model="qwen3:8b", temperature=0)

    app = build_graph(retriever_obj, llm_obj)

    if len(sys.argv) >= 3:
        question = sys.argv[2]
    else:
        question = input("请输入你的问题：")

    result = app.invoke({"question": question, "documents": [], "answer": ""})
    print("\n" + "=" * 40)
    print("问题：", question)
    print("-" * 40)
    print("答案：", result["answer"])


if __name__ == "__main__":
    main()
