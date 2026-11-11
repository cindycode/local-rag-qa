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
ollama serve                   # 在一个独立终端保持运行（macOS 首次会弹授权，点允许）
ollama pull qwen3:8b          # 生成模型（约 5GB，首次较慢）
ollama pull nomic-embed-text  # 向量化模型
```

### 3. 准备文档

建一个目录，把 `.txt` 文档丢进去（暂以 txt 起步，后续阶段扩展 PDF/Markdown）。

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

首次运行会建索引（较慢），之后复用本地 `chroma_db/`，直接问答。

> 仓库自带 `sample_docs/` 测试文档，可用它先跑通。

---

## 项目结构

```
local-rag-qa/
├── stage1-langgraph/   # LangGraph 内核：基础 RAG
│   ├── qa.py           # 核心引擎（StateGraph + retrieve/generate）
│   ├── app.py          # FastAPI 后端（/api/ingest + /api/chat）
│   ├── static/index.html  # 单页问答前端
│   ├── sample_docs/    # 测试文档
│   └── requirements.txt
├── stage2-deerflow/    # （规划中）DeerFlow 内核
├── stage3-deepseek/    # （规划中）DeepSeek Super-Agent 内核
├── stage4-codex/       # （规划中）Codex 内核
├── docs/               # 各内核的设计笔记
└── README.md
```

---

## 关注公众号

（此处放公众号二维码图，替换 `docs/qrcode.png`）

更多源码级拆解：Agent 框架的内核设计、计算机原理溯源，见公众号。

---

## License

[MIT](LICENSE)
