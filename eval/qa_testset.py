# -*- coding: utf-8 -*-
"""
local-rag-qa · 评测测试集

针对 sample_docs/ 里的三份文档（langgraph_intro / rag_basics / ollama_notes）
手写的问答对，覆盖三类问题：

1. 事实查询（单点）：答案能在一段里直接找到，测「检索能不能命中 + 生成准不准」。
2. 综合理解（跨段）：答案分散在多个片段，测「检索能不能召回全 + 生成能不能拼接」。
3. 边界问题（文档外）：文档里没有答案，测「会不会幻觉、会不会老实说不知道」。

字段说明：
- question: 问题
- ground_truth: 标准答案（用于算 answer_correctness / context_recall）
- expect_no_answer: True 表示文档里没有答案，理想回答应是「不知道/文档未提及」
"""

TESTSET = [
    # ============ 一、事实查询（单点，答案在一段里） ============
    {
        "question": "LangGraph 的核心概念叫什么？",
        "ground_truth": "LangGraph 的核心概念是 StateGraph（状态图），用节点、边、状态把应用建模成一张图。",
    },
    {
        "question": "RAG 三个字母代表什么？",
        "ground_truth": "RAG 代表 Retrieval-Augmented Generation，即检索增强生成。",
    },
    {
        "question": "本项目 Stage 1 用的向量数据库是什么？",
        "ground_truth": "本项目 Stage 1 用的向量数据库是 Qdrant（本地嵌入式模式，免 Docker）。",
    },
    {
        "question": "Ollama 是做什么的？",
        "ground_truth": "Ollama 是一个在本地运行大语言模型的工具，让你不用联网、零 API 成本就能用上 LLM。",
    },
    {
        "question": "本项目默认的生成模型和嵌入模型分别是什么？",
        "ground_truth": "生成模型是 qwen3:4b（负责写答案），嵌入模型是 bge-m3（负责把文本转成向量，中文语义更强）。",
    },
    {
        "question": "LangGraph 是谁推出的框架？",
        "ground_truth": "LangGraph 是 LangChain 团队推出的用于构建有状态、多参与者应用的低级编排框架。",
    },
    {
        "question": "LangGraph 里的「状态」是用什么定义的？",
        "ground_truth": "状态用 TypedDict 定义，整张图共享一份数据，节点只读写自己关心的字段。",
    },

    # ============ 二、综合理解（跨段，答案分散） ============
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
    {
        "question": "一个最小的 RAG 应用由哪两个节点组成？它们之间怎么传递数据？",
        "ground_truth": "一个最小 RAG 应用就是 retrieve（检索）和 generate（生成）两个节点，中间用状态传递「问题」和「检索到的文档片段」。",
    },
    {
        "question": "Ollama 有哪些常用命令？分别做什么？",
        "ground_truth": "ollama serve 启动服务；ollama pull <model> 下载模型；ollama list 查看已下载模型；ollama run <model> 进入交互式对话。",
    },

    # ============ 三、边界问题（文档外，测幻觉/诚实度） ============
    {
        "question": "LangGraph 的 checkpointer 默认存在哪个数据库？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "bge-m3 的向量维度是多少？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "Qdrant 的 HNSW 索引参数 m 和 ef 应该怎么调？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "如何给 Ollama 配置国内镜像加速下载？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "LangGraph 和 LangChain 的版本对应关系是什么？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "本项目的 Stage 2 计划用哪个框架实现？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
    {
        "question": "Ollama 支持哪些量化格式（q4_0、q8_0 等）的区别是什么？",
        "ground_truth": "",
        "expect_no_answer": True,
    },
]
