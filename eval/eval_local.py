#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local-rag-qa · 轻量评测脚本（不依赖 ragas）

为什么不用 ragas：
    ragas 0.4.x 依赖 langchain-community 里已被移除的 vertexai 模块，
    安装/导入链路脆弱。这里用「关键词召回 + LLM 裁判」自建一套轻量评测，
    全部离线、零 API 成本，且逻辑透明可解释。

评三层：
1. 检索层（硬指标，无需 LLM）—— 标准答案的关键词有多少出现在检索到的片段里。
   这是「知识库准不准」的根：检索错了，生成一定错。
2. 生成层（LLM 裁判）—— 忠实度（答案有没有被上下文支撑）+ 相关性（有没有正面回答）。
3. 边界题（诚实度）—— 文档里没有答案时，模型是否老实说「无法回答」，而不是编造。

用法：
    cd local-rag-qa/stage1-langgraph
    python ../eval/eval_local.py --docs_dir ./sample_docs

依赖：
    ollama serve 已启动，且已 pull bge-m3 和 qwen3:4b。
"""

import os
import sys
import json
import argparse
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage1-langgraph"))

from qa_testset import TESTSET
from qa import build_qa_pipeline, EMBED_MODEL, GEN_MODEL


# ---------------------------------------------------------------------------
# 从标准答案里抽「关键词」：只保留长度 >= 2 的中文词 / 英文 token，
# 去掉标点和停用词，用于判断检索片段是否覆盖了答案要点。
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "一个", "一种", "这个", "那个",
    "什么", "怎么", "为什么", "可以", "能够", "用于", "负责", "就是", "没有",
}


def extract_keywords(text: str):
    if not text:
        return []
    # 英文/数字 token + 中文连续片段
    tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]{2,}", text)
    keywords = []
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            continue
        if t in _STOPWORDS:
            continue
        keywords.append(t)
    return keywords


def retrieval_keyword_recall(keywords, documents):
    """标准答案关键词中，有多少出现在检索到的任一文档片段里。"""
    if not keywords:
        return 1.0
    haystack = " ".join(documents)
    hit = sum(1 for kw in keywords if kw in haystack)
    return hit / len(keywords)


# ---------------------------------------------------------------------------
# LLM 裁判：让 qwen3 给一个结构化 JSON 分数
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = """你是一个严格的评测裁判。请根据给定的「上下文片段」「标准答案」「模型答案」，打分。

评分规则（各 0~1 分）：
1. 忠实度（faithfulness）：模型答案里的每个事实性陈述，是否都能在上下文片段里找到依据？凭空编造=低分。
2. 相关性（relevancy）：模型答案是否正面、完整地回答了问题？答非所问=低分。

只输出一个 JSON，不要任何多余文字，格式如下：
{{"faithfulness": 0.9, "relevancy": 0.8}}

上下文片段：
{context}

标准答案：
{ground_truth}

模型答案：
{answer}
"""


def llm_judge(llm, context, ground_truth, answer):
    prompt = _JUDGE_PROMPT.format(
        context=context[:2000] or "（无上下文）",
        ground_truth=ground_truth or "（无标准答案，判模型是否诚实表示不知道）",
        answer=answer[:2000],
    )
    raw = llm.invoke(prompt).content
    # 从 LLM 输出里抠 JSON（可能被包在反引号或额外文字里）
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_no_answer(llm, answer):
    """边界题：判模型是否「诚实表示不知道」，而不是编造。"""
    prompt = (
        "判断下面这个回答，是不是「诚实地表示无法回答 / 文档里没有相关信息」。\n"
        "如果是诚实表示不知道（包含『无法回答』『没有提及』『资料不足』『未找到』等），输出 1；\n"
        "如果它编造了一个具体答案，输出 0。只输出数字。\n\n"
        f"回答：{answer}"
    )
    raw = llm.invoke(prompt).content.strip()
    m = re.search(r"[01]", raw)
    if m:
        return int(m.group(0))
    return 0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="轻量评测本地知识库（不依赖 ragas）")
    parser.add_argument("--docs_dir", default="./sample_docs", help="文档目录")
    parser.add_argument("--persist_dir", default="./qdrant_db", help="Qdrant 持久化目录")
    args = parser.parse_args()

    print(f"嵌入模型: {EMBED_MODEL} | 生成模型: {GEN_MODEL}")
    print("正在建索引...")
    graph, _ = build_qa_pipeline(args.docs_dir, args.persist_dir)

    from langchain_ollama import ChatOllama
    llm = ChatOllama(model=GEN_MODEL, temperature=0)

    results = []
    print(f"\n正在评测 {len(TESTSET)} 条问题...\n")

    for i, item in enumerate(TESTSET, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        expect_no_answer = item.get("expect_no_answer", False)

        result = graph.invoke(
            {"question": q, "documents": [], "sources": [], "answer": ""}
        )
        answer = result["answer"]
        documents = result["documents"]

        row = {"#": i, "question": q, "answer": answer[:80], "type": ""}

        if expect_no_answer:
            # 边界题：只看诚实度
            row["type"] = "边界"
            honest = judge_no_answer(llm, answer)
            row["诚实度"] = honest
            row["关键词召回"] = ""
            row["忠实度"] = ""
            row["相关性"] = ""
        else:
            row["type"] = "普通"
            keywords = extract_keywords(gt)
            recall = retrieval_keyword_recall(keywords, documents)
            row["关键词召回"] = round(recall, 2)

            scores = llm_judge(llm, "\n".join(documents), gt, answer)
            if scores:
                row["忠实度"] = round(float(scores.get("faithfulness", 0)), 2)
                row["相关性"] = round(float(scores.get("relevancy", 0)), 2)
            else:
                row["忠实度"] = "?"
                row["相关性"] = "?"
            row["诚实度"] = ""

        results.append(row)
        print(f"[{i}/{len(TESTSET)}] {q}  →  {answer[:50]}...")

    # 汇总
    print("\n" + "=" * 60)
    print("评测汇总")
    print("=" * 60)

    normal = [r for r in results if r["type"] == "普通"]
    boundary = [r for r in results if r["type"] == "边界"]

    def _avg(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else "N/A"

    if normal:
        print(f"\n【普通题 {len(normal)} 条】")
        print(f"  关键词召回率（检索准不准）: {_avg(normal, '关键词召回')}")
        print(f"  忠实度均值（会不会编造）  : {_avg(normal, '忠实度')}")
        print(f"  相关性均值（有没有答对）  : {_avg(normal, '相关性')}")

    if boundary:
        print(f"\n【边界题 {len(boundary)} 条】")
        print(f"  诚实度均值（老实说不知道 vs 编造）: {_avg(boundary, '诚实度')}")

    # 存 CSV
    import csv
    out = os.path.join(os.path.dirname(__file__), "eval_result.csv")
    fieldnames = ["#", "type", "question", "answer", "关键词召回", "忠实度", "相关性", "诚实度"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"\n明细已存到 {out}")


if __name__ == "__main__":
    main()
