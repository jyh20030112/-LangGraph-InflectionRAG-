import os
import time
from typing import Any, Dict, List
from markitdown import MarkItDown
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()


class RAGTOOL:
    def __init__(self, file_path: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.current_document = None
        self.stats = {"documents_loaded": 0}
        self._md_instance: MarkItDown | None = None

        self.embedding_api_key = os.environ.get("API_KEY")
        self.embedding_base_url = os.environ.get("EMBEDDING_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.embedding_model = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")

        self.qdrant_url = os.environ.get("QDRANT_URL")
        self.qdrant_api_key = os.environ.get("QDRANT_API_KEY")
        self.qdrant_collection = os.environ.get("QDRANT_COLLECTION", "RAG_SEVER")

        self._embedding_client: OpenAI | None = None
        self._qdrant_client: QdrantClient | None = None

    # ---------- 核心：MarkItDown 单例 ----------
    def _get_markitdown_instance(self) -> MarkItDown | None:
        """延迟创建 MarkItDown 实例（避免重复初始化）"""
        if self._md_instance is None:
            try:
                self._md_instance = MarkItDown()
                print("[RAG] MarkItDown 实例创建成功")
            except Exception as e:
                print(f"[ERROR] MarkItDown 初始化失败: {e}")
                return None
        return self._md_instance

    # ---------- PDF 增强处理 ----------
    def _enhanced_pdf_processing(self, path: str) -> str:
        """
        PDF 增强解析：
        1. 先用 MarkItDown 转 Markdown
        2. 失败则降级为纯文本读取
        """
        md = self._get_markitdown_instance()
        if md is None:
            return self._fallback_text_reader(path)

        try:
            result = md.convert(path)
            text = getattr(result, "text_content", None)
            if isinstance(text, str) and text.strip():
                print(f"[RAG] PDF增强解析成功: {path} -> {len(text)} chars Markdown")
                return text
        except Exception as e:
            print(f"[WARNING] MarkItDown PDF解析失败 {path}: {e}")

        return self._fallback_text_reader(path)

    # ---------- 统一文档转换入口 ----------
    def _convert_to_markdown(self, path: str) -> str:
        """将任意格式文档转换为 Markdown 文本"""
        if not os.path.exists(path):
            print(f"[ERROR] 文件不存在: {path}")
            return ""

        ext = (os.path.splitext(path)[1] or "").lower()

        # PDF 走增强处理
        if ext == ".pdf":
            return self._enhanced_pdf_processing(path)

        # 其他格式走通用 MarkItDown
        md = self._get_markitdown_instance()
        if md is None:
            return self._fallback_text_reader(path)

        try:
            result = md.convert(path)
            text = getattr(result, "text_content", None)
            if isinstance(text, str) and text.strip():
                print(f"[RAG] MarkItDown转换成功: {path} -> {len(text)} chars Markdown")
                return text
            return ""
        except Exception as e:
            print(f"[WARNING] MarkItDown转换失败 {path}: {e}")
            return self._fallback_text_reader(path)

    # ---------- 降级方案 ----------
    def _fallback_text_reader(self, path: str) -> str:
        """当 MarkItDown 失败时，尝试直接读取为文本"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    print(f"[RAG] 降级文本读取成功: {path} -> {len(content)} chars")
                    return content
        except Exception as e:
            print(f"[ERROR] 降级读取也失败 {path}: {e}")
        return ""

    def _get_embedding_client(self) -> OpenAI | None:
        if self._embedding_client is None:
            if not self.embedding_api_key:
                print("[ERROR] 未配置 EMBEDDING_MODEl 环境变量")
                return None
            try:
                self._embedding_client = OpenAI(
                    api_key=self.embedding_api_key,
                    base_url=self.embedding_base_url,
                )
                print("[RAG] Embedding 客户端创建成功")
            except Exception as e:
                print(f"[ERROR] Embedding 客户端初始化失败: {e}")
                return None
        return self._embedding_client

    def _get_qdrant_client(self) -> QdrantClient | None:
        if self._qdrant_client is None:
            if not self.qdrant_url:
                print("[ERROR] 未配置 QDRANT_URL 环境变量")
                return None
            try:
                self._qdrant_client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60,
                )
                print("[RAG] Qdrant 客户端创建成功")
            except Exception as e:
                print(f"[ERROR] Qdrant 客户端初始化失败: {e}")
                return None
        return self._qdrant_client

    def _ensure_collection(self, vector_size: int) -> bool:
        qdrant = self._get_qdrant_client()
        if qdrant is None:
            return False

        collections = [c.name for c in qdrant.get_collections().collections]
        if self.qdrant_collection not in collections:
            qdrant.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"[RAG] Qdrant 集合已创建: {self.qdrant_collection} (dim={vector_size})")
        return True

    def _BATCH_SIZE(self) -> int:
        return 10

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._get_embedding_client()
        if client is None:
            raise RuntimeError("Embedding 客户端不可用")

        batch_size = self._BATCH_SIZE()
        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            response = client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)
            print(f"[RAG] 向量化进度: {min(batch_start + batch_size, len(texts))}/{len(texts)}")

        return all_embeddings

    def _embed_chunks(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0

        client = self._get_embedding_client()
        if client is None:
            print("[RAG] 跳过向量化：Embedding 客户端不可用")
            return 0

        qdrant = self._get_qdrant_client()
        if qdrant is None:
            print("[RAG] 跳过存储：Qdrant 客户端不可用")
            return 0

        texts = [c["content"] for c in chunks]
        print(f"[RAG] 开始向量化 {len(texts)} 个 chunk...")

        vectors = self._embed_texts(texts)
        vector_size = len(vectors[0])

        if not self._ensure_collection(vector_size):
            print("[ERROR] Qdrant 集合创建失败")
            return 0

        points = []
        stored_total = 0
        upsert_batch = 20
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {
                "content": chunk["content"],
                "source": os.path.basename(self.file_path),
            }
            if chunk.get("heading_path"):
                payload["heading_path"] = chunk["heading_path"]
            if chunk.get("start") is not None:
                payload["start"] = chunk["start"]
            if chunk.get("end") is not None:
                payload["end"] = chunk["end"]

            points.append(PointStruct(
                id=i,
                vector=vec,
                payload=payload,
            ))

            if len(points) >= upsert_batch or i == len(chunks) - 1:
                qdrant.upsert(
                    collection_name=self.qdrant_collection,
                    points=points,
                )
                stored_total += len(points)
                print(f"[RAG] 已写入 Qdrant: {stored_total}/{len(chunks)}")
                points = []

        print(f"[RAG] 向量化与存储完成，{stored_total} 个向量已写入 Qdrant")
        return stored_total

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        client = self._get_embedding_client()
        if client is None:
            print("[ERROR] Embedding 客户端不可用，无法检索")
            return []

        qdrant = self._get_qdrant_client()
        if qdrant is None:
            print("[ERROR] Qdrant 客户端不可用，无法检索")
            return []

        query_vector = self._embed_texts([query])[0]

        response = qdrant.query_points(
            collection_name=self.qdrant_collection,
            query=query_vector,
            limit=top_k,
        )

        return [
            {
                "content": r.payload.get("content", ""),
                "source": r.payload.get("source", ""),
                "heading_path": r.payload.get("heading_path"),
                "score": r.score,
            }
            for r in response.points
        ]

    # ---------- 完整 RAG 处理流水线 ----------
    def rag_tool(self) -> Dict[str, Any]:
        """核心流水线：转换 → 分块 → 向量化"""
        # 第一步：MarkItDown 转换
        markdown_text = self._convert_to_markdown(self.file_path)
        if not markdown_text:
            return {"success": False, "error": "文档转换失败，无法提取文本"}

        print(f"[RAG] 文档转换完成，共 {len(markdown_text)} 字符")

        # 第二步：智能分块（按标题切段 → token 分块）
        paragraphs = self._split_paragraphs_with_headings(markdown_text)
        chunks = self._chunk_paragraphs(paragraphs, self.chunk_size, self.chunk_overlap)
        print(f"[RAG] 分块完成，共 {len(chunks)} 块")

        # 第三步：向量化并存入 Qdrant
        stored_count = self._embed_chunks(chunks)

        return {
            "success": True,
            "chunks_count": len(chunks),
            "vectors_stored": stored_count,
        }

    # ---------- 智能分块 ----------
    @staticmethod
    def _approx_token_len(text: str) -> int:
        """近似估计Token长度，支持中英文混合"""
        def _is_cjk(ch: str) -> bool:
            """判断是否为CJK字符"""
            code = ord(ch)
            return (
                0x4E00 <= code <= 0x9FFF or  # CJK统一汉字
                0x3400 <= code <= 0x4DBF or  # CJK扩展A
                0x20000 <= code <= 0x2A6DF or # CJK扩展B
                0x2A700 <= code <= 0x2B73F or # CJK扩展C
                0x2B740 <= code <= 0x2B81F or # CJK扩展D
                0x2B820 <= code <= 0x2CEAF or # CJK扩展E
                0xF900 <= code <= 0xFAFF      # CJK兼容汉字
            )
        # CJK字符按1 token计算
        cjk = sum(1 for ch in text if _is_cjk(ch))
        # 其他字符按空白分词计算
        non_cjk_tokens = len([t for t in text.split() if t])
        return cjk + non_cjk_tokens

    def _split_paragraphs_with_headings(self, text: str) -> List[Dict]:
        """根据标题层次分割段落，保持语义完整性"""
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0

        def flush_buf(end_pos: int):
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append({
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            })

        for ln in lines:
            raw = ln
            if raw.strip().startswith("#"):
                flush_buf(char_pos)
                level = len(raw) - len(raw.lstrip("#"))
                title = raw.lstrip("#").strip()

                if level <= 0:
                    level = 1
                if level <= len(heading_stack):
                    heading_stack = heading_stack[:level - 1]
                heading_stack.append(title)

                char_pos += len(raw) + 1
                continue

            if raw.strip() == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            char_pos += len(raw) + 1

        flush_buf(char_pos)

        if not paragraphs:
            paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]

        return paragraphs

    def _chunk_paragraphs(self, paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
        """基于Token数量的智能分块"""
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0

        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = self._approx_token_len(p["content"]) or 1

            if cur_tokens + p_tokens <= chunk_tokens or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                content = "\n\n".join(x["content"] for x in cur)
                start = cur[0]["start"]
                end = cur[-1]["end"]
                heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

                chunks.append({
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                })

                if overlap_tokens > 0 and cur:
                    kept: List[Dict] = []
                    kept_tokens = 0
                    for x in reversed(cur):
                        t = self._approx_token_len(x["content"]) or 1
                        if kept_tokens + t > overlap_tokens:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens
                else:
                    cur = []
                    cur_tokens = 0

        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)

            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })

        return chunks

    # ---------- 对外入口 ----------
    def load_document(self) -> Dict[str, Any]:
        """加载 PDF 文档到知识库"""
        if not os.path.exists(self.file_path):
            return {"success": False, "message": f"文件不存在: {self.file_path}"}

        start_time = time.time()

        result = self.rag_tool()

        process_time = time.time() - start_time

        if result.get("success"):
            self.current_document = os.path.basename(self.file_path)
            self.stats["documents_loaded"] += 1
            return {
                "success": True,
                "message": f"加载成功！(耗时: {process_time:.1f}秒)",
                "document": self.current_document,
                "chunks_count": result.get("chunks_count", 0),
                "vectors_stored": result.get("vectors_stored", 0),
            }
        else:
            return {
                "success": False,
                "message": f"加载失败: {result.get('error', '未知错误')}",
            }

def main():
    rag = RAGTOOL()
    # result = rag.load_document()
    # print(result)
    results = rag.search('transfomer是什么？', top_k=1)
    for r in results:
        print(f"[score={r['score']:.3f}] [{r['heading_path']}] {r['content'][:]}...")

if __name__ == '__main__':
    main()