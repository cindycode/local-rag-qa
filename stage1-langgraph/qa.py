#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · Stage 1 · LangGraph 内核

同一个「本地知识库问答」，用不同 Agent 框架各实现一遍。
这是第一阶段：用 LangGraph 的 StateGraph 串起 RAG 流水线。

存储与模型选型（本地全离线，零 API 成本）：
- 向量库：Qdrant（本地嵌入式模式，QdrantClient(path=...)，免 Docker）
- 嵌入模型：bge-m3（BAAI 智源，中文语义向量标杆）
- 生成模型：qwen3:4b（千问，中文问答强；网络受限时用 4b，可换 8b 效果更好）

本文件是「核心引擎」，同时服务两种入口：
1. CLI：python qa.py <文档目录> [问题]
2. Web：app.py 里的 FastAPI 接口调用 build_qa_pipeline()

内核要暴露的点（LangGraph 的设计）：
1. StateGraph —— 把「问答」建模成一张有状态的图，节点是步骤，边是流转。
2. TypedDict 状态 —— 整张图共享一份带类型的状态，节点只改自己关心的字段。
3. 显式节点 / 边 —— retrieve 和 generate 是两个独立节点，数据靠状态传递。
4. 可扩展 —— 之后想加「重写问题」「多路召回」「重排序」，都是往图里加节点。

本阶段额外暴露的「存储内核」：
- Qdrant 的 payload 元数据：每个向量块都带上来源、内容，检索时能按元数据过滤。
- 混合检索（hybrid）：向量召回 + 关键词过滤，弥补纯向量对「专有名词/数字」的召回短板。
"""

from typing import TypedDict, List, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------------------------
# 配置（本地离线默认值，可用环境变量覆盖）
# ---------------------------------------------------------------------------
EMBED_MODEL = "bge-m3"          # 嵌入模型：中文语义向量
GEN_MODEL = "qwen3:4b"          # 生成模型：中文问答（网络受限时用 4b，效果略降但下载快）
CHUNK_SIZE = 500                # 切块大小
CHUNK_OVERLAP = 50              # 切块重叠
TOP_K = 3                       # 检索召回条数
COLLECTION_NAME = "local_rag"   # Qdrant 集合名


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
#    从 Qdrant 向量库里召回和问题最相关的片段
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
# 5. 索引构建：本地文档切块、向量化、存入 Qdrant（本地嵌入式）
# ---------------------------------------------------------------------------
def _load_documents(docs_dir: str):
    """加载 txt / md / pdf 三种格式，合并成统一文档列表。

    多格式支持是「真知识库」的第一步：读者的资料不会全是 txt，
    通常混着 PDF（论文/报告）和 Markdown（笔记）。
    """
    docs = []
    # txt + md 走文本 loader
    txt_loader = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=False
    )
    md_loader = DirectoryLoader(
        docs_dir, glob="**/*.md", loader_cls=TextLoader, show_progress=False
    )
    docs += txt_loader.load()
    docs += md_loader.load()

    # pdf 走 PyPDFLoader（需要 pypdf 依赖）
    pdf_loader = DirectoryLoader(
        docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False
    )
    try:
        docs += pdf_loader.load()
    except Exception as e:
        # 没装 pypdf 或目录无 pdf 时，跳过并提示，不阻塞 txt/md
        print(f"  [提示] PDF 加载跳过：{e}（如需要 PDF 支持，pip install pypdf）")

    return docs


def build_index(docs_dir: str, persist_dir: str = "./qdrant_db"):
    docs = _load_documents(docs_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 本地嵌入式 Qdrant：数据直接落在 persist_dir，免 Docker
    client = QdrantClient(path=persist_dir)

    # 若集合已存在，先重建（保证每次 ingest 都是全量新鲜索引）
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1024,  # bge-m3 输出 1024 维
            distance=Distance.COSINE,
        ),
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    vectorstore.add_documents(chunks)
    return vectorstore


# ---------------------------------------------------------------------------
# 6. 组装完整流水线：返回 (compiled_graph, vectorstore)
# ---------------------------------------------------------------------------
def build_qa_pipeline(
    docs_dir: str,
    persist_dir: str = "./qdrant_db",
    model: str = GEN_MODEL,
    embed_model: str = EMBED_MODEL,
):
    """给 Web 后端用：建索引 + 组装图，返回 (graph, vectorstore)。

    说明：bge-m3 的向量维度是 1024，建索引时写死在 build_index 里；
    若换嵌入模型需同步改 VectorParams 的 size。
    """
    vectorstore = build_index(docs_dir, persist_dir)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=model, temperature=0)
    graph = build_graph(retriever, llm)
    return graph, vectorstore


# ---------------------------------------------------------------------------
# 7. 按元数据过滤检索（Stage 1 存储层卖点：filter 下推）
#    在 Qdrant 上按来源文件名过滤，再做向量召回。
#    这是「混合检索」的第一步：先学会「约束检索范围」，再谈「多路召回融合」。
# ---------------------------------------------------------------------------
def search_with_filter(
    vectorstore,
    query: str,
    source_contains: Optional[str] = None,
    k: int = TOP_K,
):
    """按来源过滤检索。source_contains 为空则退化为普通向量检索。

    Qdrant 的 filter 是「下推」到服务端执行的：不是把全部结果拉回来再筛，
    而是在向量搜索的同时就只召回满足条件的点。这是它相比「先全量检索再内存过滤」
    的性能优势，也是「检索时约束」这个进阶主题的入口。
    """
    from qdrant_client.models import Filter, FieldCondition, MatchText

    search_kwargs = {"k": k}
    if source_contains:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchText(text=source_contains),
                )
            ]
        )
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    return retriever.invoke(query)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main():
    import sys

    if len(sys.argv) < 2:
        print("用法：python qa.py <文档目录> [问题]")
        print("  第一次运行会先建索引，之后直接问答。")
        print("  依赖：ollama serve 已启动，且已 pull bge-m3 和 qwen3:4b。")
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
