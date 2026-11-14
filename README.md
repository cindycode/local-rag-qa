# local-rag-qa

> 同一个「本地知识库问答」，用四个 Agent 框架各实现一遍，看每个内核的深浅。

把你的 PDF、笔记、公众号历史喂进去，在本地（零 API 成本）跑一个能问答的知识库。
这不是一个「玩具 demo」，而是一个**刻意用四种内核重写同一功能**的项目：每一阶段换一个框架，
借那个框架最核心的设计，把同一个 RAG 问答做出不同的「内核气质」。

---

## 为什么做这个

大多数 RAG 教程只教你「怎么用 LangChain 搭一个问答」，但很少有人回答：

- 同一件事，用 **LangGraph / DeerFlow / DeepSeek Super-Agent / Codex** 分别怎么写？
- 它们的「状态」「编排」「执行」模型，落到同一个功能上，差异到底在哪？
- 为什么有的框架把流程画成「图」，有的画成「循环」，有的画成「副作用队列」？

这个仓库就是用四个内核回答这些问题。每换一个内核，功能就顺着那个框架的强项往前走一步，
从而**用「加功能」逼出「内核」**。

---

## 四阶段路线图

| 阶段 | 内核 | 暴露的内核设计 | 新增功能 |
|---|---|---|---|
| Stage 1 | **LangGraph** | StateGraph / 状态 / 显式节点边 | 基础 RAG：建索引 → 检索 → 生成 |
| Stage 2 | **DeerFlow** | Agent Harness / 规划-执行循环 | 多步问答：拆解问题、分步检索 |
| Stage 3 | **DeepSeek Super-Agent** | Effects 队列 / Outbox 幂等 | 可靠副作用：索引写入不丢不重 |
| Stage 4 | **Codex** | 模型路由 / 执行沙箱 | 多模型切换、可插拔后端 |

---

## 快速开始（Stage 1 · LangGraph）

### 1. 装依赖

```bash
pip install -r stage1-langgraph/requirements.txt
```

### 2. 启动本地 Ollama 并拉模型

```bash
ollama serve            # 在一个独立终端保持运行（macOS 首次会弹授权，点允许）
ollama pull qwen3:4b   # 生成模型（约 2.4GB，负责写答案；网络好可换 qwen3:8b 效果更佳）
ollama pull bge-m3     # 嵌入模型（约 1GB，负责把文本转成向量，中文语义最强）
```

> 选型说明：嵌入模型用 **bge-m3**（BAAI 智源，中文语义向量标杆），而非默认的
> nomic-embed-text，因为本项目文档和提问都是中文，bge-m3 的召回质量明显更好。
> 向量库用 **Qdrant**（本地嵌入式模式，免 Docker），支持按元数据过滤，是后续
> 「混合检索」升级的基础。

### 3. 准备文档

建一个目录，把 `.txt`、`.md`、`.pdf` 文档丢进去（支持三种格式；PDF 需要 `pypdf` 依赖，已在 requirements 里）。

### 4. 建索引 + 问答

**方式 A：Web 界面（推荐）**

```bash
cd stage1-langgraph
uvicorn app:app --reload --port 8000
```

浏览器打开 http://127.0.0.1:8000，在页面上填文档目录、建索引、提问。
答案下方会展示「检索到的证据片段」，这是 RAG 与普通聊天最大的区别：答案有据可查。

**方式 B：命令行**

```bash
cd stage1-langgraph
python qa.py /path/to/your/docs "你的问题"
```

首次运行会建索引（较慢），之后复用本地 `qdrant_db/`，直接问答。

> 仓库自带 `sample_docs/` 测试文档，可用它先跑通。

---

## 验证准确性（轻量评测，不依赖 ragas）

用本地 qwen3 做 LLM-as-Judge，给知识库打三层分（0~1，越高越好）：

| 层 | 指标 | 含义 |
|---|---|---|
| 检索层 | 关键词召回率 | 标准答案的关键词有多少出现在检索到的片段里（硬指标，测「找得准不准」） |
| 生成层 | 忠实度 / 相关性 | 答案有没有被上下文支撑（测幻觉）、有没有正面回答问题（qwen3 裁判） |
| 边界题 | 诚实度 | 文档里没答案时，模型是否老实说「无法回答」而不是编造 |

```bash
# 跑评测（Ollama 保持运行）
cd stage1-langgraph
python ../eval/eval_local.py --docs_dir ./sample_docs
```

结果会打印每道题分数 + 分类汇总（普通题 / 边界题），并落到 `eval/eval_result.csv`。
**四阶段换内核时各跑一遍，就能横向对比「哪个内核检索/生成更准」**——这是本仓库「用评测逼出内核优劣」的核心。

> 说明：曾尝试用 [RAGAS](https://docs.ragas.io)，但它 0.4.x 依赖 langchain-community
> 里已被移除的 `vertexai` 模块，安装/导入链路脆弱。故改用自建的轻量评测
> （`eval/eval_local.py`），零第三方评测依赖、逻辑透明、还能自定义「边界题诚实度」这类
> 通用库给不了的指标。

---

## 项目结构

```
local-rag-qa/
├── stage1-langgraph/   # LangGraph 内核：基础 RAG
│   ├── qa.py           # 核心引擎（StateGraph + retrieve/generate + Qdrant 存储）
│   ├── app.py          # FastAPI 后端（/api/ingest + /api/chat）
│   ├── static/index.html  # 单页问答前端
│   ├── sample_docs/    # 测试文档
│   └── requirements.txt
├── stage2-deerflow/    # （规划中）DeerFlow 内核
├── stage3-deepseek/    # （规划中）DeepSeek Super-Agent 内核
├── stage4-codex/       # （规划中）Codex 内核
├── docs/               # 各内核的设计笔记
├── eval/               # 评测（qa_testset.py 测试集 + eval_local.py 轻量评测脚本）
└── README.md
```

---

## 关注公众号

（此处放公众号二维码图，替换 `docs/qrcode.png`）

更多源码级拆解：Agent 框架的内核设计、计算机原理溯源，见公众号。

---

## License

[MIT](LICENSE)
