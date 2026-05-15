# ReflectionRAG

基于 **LangGraph** 构建的自适应 RAG 智能问答系统，集成 **通义千问 (Qwen)** 大模型、**Qdrant** 向量数据库与 **Tavily** 网络搜索，实现文档解析 → 智能分块 → 向量化存储 → 问题路由 → 检索增强生成的全链路闭环。

---

## 项目特点

- 🔀 **智能路由**：LLM 自动判断走本地向量库检索还是网络搜索
- 🔄 **自适应反馈闭环**：检索文档不相关时自动重写 Query 并重新检索
- ✅ **文档相关性评分**：LLM 对每篇检索文档进行 yes/no 评分，过滤噪声
- 🛡️ **检索失败降级**：检索或搜索失败时自动切换原始大模型直接回答
- 📄 **多格式文档解析**：基于 MarkItDown 统一处理 PDF/Word/Excel/PPT
- 🧠 **语义感知分块**：Markdown 标题层级识别 + Token 估算自适应分块
- 📊 **丰富元数据**：每个 Chunk 附带章节路径、来源文件和位置信息

---

## 工作流

```
用户问题
   │
   ▼
┌──────────┐
│ 问题路由  │ ◄── LLM 判断：本地 RAG 还是网络搜索
└────┬─────┘
     │
  ┌──┴──────────┐
  ▼             ▼
┌──────┐    ┌──────────┐
│本地检索│    │ 网络搜索  │ ◄── Tavily Search API
└──┬───┘    └────┬─────┘
   │             │
   ▼             │
┌──────────┐     │
│ 文档评分  │     │
└────┬─────┘     │
     │           │
  ┌──┴───┐      │
  ▼      ▼      │
 yes    no      │
  │      │      │
  │   ┌──────────────┐
  │   │ 问题重写/改写  │ ◄── 优化 Query 后重新检索
  │   └──────┬───────┘
  │          │
  │    ┌─────┘ (循环)
  │    │
  ▼    ▼
┌──────────┐
│ 大模型生成 │ ◄── RAG 上下文 + LLM
└────┬─────┘
     │
  ┌──┴──────┐
  ▼         │
 yes       no
  │         │
  ▼         ▼
RAG回答   降级回答 ◄── 原始模型直接回答
```

---

## 项目结构

```
.
├── graph.py              # LangGraph 工作流编排
├── router.py             # LLM 链定义（路由/评分/重写/生成/降级）
├── RAGtool.py            # 文档解析 → 分块 → 向量化 → 存储 → 检索
├── requirements.txt      # 依赖清单
├── test.py               # 测试脚本
├── .env                  # 环境变量（需自行创建）
└── pdf_file/             # 待索引的 PDF 文件
```

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/jyh20030112/ReflectionRAG.git
cd ReflectionRAG
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
API_KEY=sk-your-dashscope-key
EMBEDDING_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
CHAT_MODEL=qwen3-max

QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-key
QDRANT_COLLECTION=RAG_SEVER

TAVILY_API_KEY=tvly-your-tavily-key
```

### 3. 上传文档并写入向量库

```python
from RAGtool import RAGTOOL

tool = RAGTOOL(file_path="./pdf_file/your-document.pdf")
result = tool.load_document()
print(result)
```

### 4. 启动问答

```python
from graph import build_app

app = build_app()
result = app.invoke({"question": "大语言模型的底层机制是什么？"})
print(result["generation"])
```

---

## 核心模块

### graph.py — LangGraph 工作流

| 节点 | 功能 |
|------|------|
| `route` | LLM 判断问题类型，决定走 RAG 还是 Web Search |
| `vectorestore` | 调用 Qdrant 向量检索 |
| `web_search` | 调用 Tavily 网络搜索 |
| `grade_documents` | LLM 逐一评分文档相关性 |
| `transform_query` | 文档全不相关时重写 Query |
| `generation` | 基于上下文调用大模型生成答案 |

### router.py — LLM 链

| 链 | 用途 |
|----|------|
| `bulid_question_router()` | 结构化输出路由判断 (vectorstore/web_search) |
| `build_retrieval_grader()` | 结构化输出文档相关性评分 (yes/no) |
| `build_question_rewriter()` | 查询重写优化 |
| `build_rag_chain()` | RAG 上下文增强生成 |
| `build_model_chain()` | 降级时原始模型直接回答 |

### RAGtool.py — 文档处理流水线

| 方法 | 功能 |
|------|------|
| `_enhanced_pdf_processing()` | MarkItDown PDF 增强解析 |
| `_split_paragraphs_with_headings()` | 按 Markdown 标题分段 |
| `_chunk_paragraphs()` | Token 估算自适应分块 |
| `_embed_chunks()` | 批量向量化 + Qdrant 分批写入 |
| `search()` | 语义检索返回 Top-K 结果 |

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 工作流编排 | LangGraph |
| LLM 框架 | LangChain |
| 大模型 | 通义千问 Qwen (DashScope API) |
| 向量数据库 | Qdrant Cloud |
| Embedding | DashScope text-embedding-v3 |
| 网络搜索 | Tavily Search API |
| 文档解析 | MarkItDown |
| 环境管理 | python-dotenv |

---

## License

MIT
