# -*- coding: utf-8 -*-
"""
local-rag-qa · 评测测试集

针对 sample_docs/ 里的三份文档（langgraph_intro / rag_basics / ollama_notes）
手写的问答对，覆盖两类问题：

1. 事实查询（单点）：答案能在一段里直接找到，测「检索能不能命中 + 生成准不准」。
2. 综合理解（跨段）：答案分散在多个片段，测「检索能不能召回全 + 生成能不能拼接」。

字段说明（RAGAS 格式）：
- question: 问题
- ground_truth: 标准答案（用于算 answer_correctness / context_recall）
"""

TESTSET = [
    # ---------- 事实查询 ----------
    {
        "question": "LangGraph 的核心概念叫什么？",
        "ground_truth": "LangGraph 的核心概念是 StateGraph（状态图），用节点、边、状态把应用建模成一张图。",
    },
    {
        "question": "RAG 三个字母代表什么？",
        "ground_truth": "RAG 代表 Retrieval-Augmented Generation，检索增强生成。",
    },
    {
        "question": "本项目 Stage 1 用的向量数据库是什么？",
        "ground_truth": "本项目 Stage 1 用的向量数据库是 Chroma（注：已升级为 Qdrant）。",
    },
    {
        "question": "Ollama 是做什么的？",
        "ground_truth": "Ollama 是一个在本地运行大语言模型的工具，让你不用联网、零 API 成本就能用上 LLM。",
    },
    # ---------- 综合理解 ----------
    {
        "question": "RAG 的工作流程分哪三步？",
        "ground_truth": "RAG 分三步：索引（切块+向量化+入库）、检索（问题转向量找最相似块）、生成（把检索结果拼进提示词让 LLM 作答）。",
    },
    {
        "question": "LangGraph 相比传统链式（Chain）有哪些优势？",
        "ground_truth": "LangGraph 支持循环（ReAct 思考-行动-观察）、分支（条件边）、持久化（checkpointer，可中断恢复回溯）。",
    },
    {
        "question": "为什么嵌入模型和生成模型要分开？",
        "ground_truth": "因为任务不同：嵌入模型追求语义相似，生成模型追求会说话。",
    },
    {
        "question": "RAG 解决了大模型的哪两个关键问题？",
        "ground_truth": "RAG 解决两个问题：幻觉（只根据检索上下文回答不编造）和知识时效（知识库可随时更新无需重训模型）。",
    },
]
