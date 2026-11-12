#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · RAGAS 评测脚本

验证知识库的「准确性」：对测试集里的每个问题，真实跑一遍
「检索 → 生成」，再用 RAGAS 的 LLM-as-Judge 打分。

四指标（0~1，越高越好）：
- faithfulness        答案忠实度：答案有没有被检索到的上下文支持（测幻觉）
- answer_relevancy    答案相关性：答案有没有正面回答问题
- context_precision   上下文精度：检索到的片段里，相关的占比高不高
- context_recall      上下文召回：该召回的相关片段，都召回了没有

用法：
    cd local-rag-qa/stage1-langgraph
    python ../eval/eval_qa.py --docs_dir ./sample_docs

依赖：
    ollama serve 已启动，且已 pull bge-m3 和 qwen3:4b（零 API 成本）。
"""

import os
import sys
import argparse

# 让 eval 目录能 import 到上一层的 qa.py（stage1-langgraph/qa.py）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1-langgraph"))

from datasets import Dataset

from qa_testset import TESTSET
from qa import build_qa_pipeline, EMBED_MODEL, GEN_MODEL


def run_retrieval_and_generate(graph, question: str):
    """对单个问题，跑图得到答案 + 检索上下文。"""
    result = graph.invoke(
        {"question": question, "documents": [], "sources": [], "answer": ""}
    )
    return result["answer"], result["documents"]


def build_eval_dataset(graph):
    """把测试集跑成 RAGAS 需要的 Dataset。

    RAGAS 字段：question / answer / contexts(复数) / ground_truth
    """
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in TESTSET:
        q = item["question"]
        answer, contexts = run_retrieval_and_generate(graph, q)
        rows["question"].append(q)
        rows["answer"].append(answer)
        rows["contexts"].append(contexts)  # 注意是 list[list[str]]
        rows["ground_truth"].append(item["ground_truth"])
    return Dataset.from_dict(rows)


def main():
    parser = argparse.ArgumentParser(description="RAGAS 评测本地知识库")
    parser.add_argument("--docs_dir", default="./sample_docs", help="文档目录")
    parser.add_argument("--persist_dir", default="./qdrant_db", help="Qdrant 持久化目录")
    args = parser.parse_args()

    print(f"嵌入模型: {EMBED_MODEL} | 生成模型: {GEN_MODEL}")
    print("正在建索引...")
    graph, _ = build_qa_pipeline(args.docs_dir, args.persist_dir)

    print(f"正在跑 {len(TESTSET)} 条测试...")
    dataset = build_eval_dataset(graph)

    # 用本地 Ollama 当裁判（零成本），嵌入用 bge-m3
    from langchain_ollama import ChatOllama
    from langchain_ollama import OllamaEmbeddings
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    llm = ChatOllama(model=GEN_MODEL, temperature=0)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "=" * 50)
    print("评测结果（0~1，越高越好）")
    print("=" * 50)
    df = result.to_pandas()
    # 打印每道题的分数 + 平均分
    print(df.round(3).to_string(index=False))
    print("\n--- 平均分 ---")
    print(df.mean(numeric_only=True).round(3).to_string())

    # 存成 CSV 方便对比（四阶段换内核时，各阶段跑一遍就能横向对比）
    out = os.path.join(os.path.dirname(__file__), "ragas_result.csv")
    df.to_csv(out, index=False)
    print(f"\n结果已存到 {out}")


if __name__ == "__main__":
    main()
