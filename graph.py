import os
from RAGtool import RAGTOOL
from router import bulid_question_router, build_rag_chain, build_model_chain, build_retrieval_grader, build_question_rewriter
from typing import TypedDict
from tavily import TavilyClient

from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph

from dotenv import load_dotenv
load_dotenv()

class GraphState(TypedDict, total=False):
    '''
    定义全局状态
    type: 决定是RAG还是网络
    question: 问题
    generation: 答案
    documents: 文件文档
    '''
    type: str
    question: str
    generation: str
    documents: list[Document]

class InflectionRAG:
    def __init__(self) -> None:
        self.question_router = bulid_question_router()
        self.rag_chain = build_rag_chain()
        self.model_chain = build_model_chain()
        self.retrieval_grader = build_retrieval_grader()
        self.question_rewriter = build_question_rewriter()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.rag_tool = RAGTOOL()
        
    def print(self, state: GraphState):
        # print(state['type'])
        # print(state['question'])
        # print(state['documents'])
        print(state['generation'])
    
    def transform_query(self, state: GraphState) -> GraphState:
        print("---问题改写---")
        question = state["question"]
        better_question = self.question_rewriter.invoke({"question": question})
        return {"question": better_question}

    def decide_to_generate(self, state: GraphState) -> str:
        print("---检查是否有匹配文档---")
        filtered_documents = state["documents"]
        if not filtered_documents:
            print("---决定: 所有的文档和问题都不匹配，问题改写！！---")
            return "transform_query"
        print("---决定: 可正常生成---")
        return "generation"

    def grade_documents(self, state: GraphState) -> dict:
        print("---检查问题是否和检索文档相关---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for doc in documents:
            result = self.retrieval_grader.invoke({
                "document": doc.page_content,
                "question": question,
            })
            if result.binary_score == "yes":
                print(f"文档相关，保留")
                filtered_docs.append(doc)
            else:
                print(f"文档不相关，丢弃")

        if not filtered_docs:
            print("[WARN] 所有文档均不相关")

        print(f"  过滤结果: {len(filtered_docs)}/{len(documents)} 个文档通过相关性检查")
        return {"documents": filtered_docs}


    def generation(self, state: GraphState):
        print("---大模型调用生成---")
        try:
            question = state["question"]
            documents = state["documents"]

            first_doc = documents[0].page_content
            if first_doc.startswith("搜索失败"):
                print("[WARN] 检索失败，降级为原始模型直接回答")
                generation = self.model_chain.invoke({"question": question})
                return {"generation": generation}

            format_docs = "\n\n".join(doc.page_content for doc in documents)
            generation = self.rag_chain.invoke({"context": format_docs, "question": question})
            return {"documents": documents, "question": question, "generation": generation}
        except Exception as e:
            print(f"[ERROR] 大模型调用失败: {e}")
            return {"generation": f"生成失败: {e}"}


    def rag_search(self, state: GraphState) -> dict:
        try:
            results = self.rag_tool.search(
                query=state["question"],
                top_k=5,
            )
            raw_results = "\n\n".join(
                f"[score={r['score']:.3f}] {r.get('heading_path', '')}\n{r['content']}"
                for r in results
            )
            return {"documents": [Document(page_content=raw_results)]}
        except Exception as e:
            print(f"[ERROR] 本地RAG 搜索失败: {e}")
            return {"documents": [Document(page_content=f"搜索失败: {e}")]}



    def tavily_search(self, state: GraphState) -> dict:
        try:
            print(f"网络搜索: {state['question']}")
            response = self.tavily_client.search(
                query=state["question"],
                search_depth="basic",
                max_results=5,
            )
            raw_results = response.get("results", [])
            search_results = "\n\n".join(
                f"[{r.get('title', '')}]({r.get('url', '')})\n{r.get('content', '')}"
                for r in raw_results
            )
            return {"documents": [Document(page_content=search_results)]}
        except Exception as e:
            print(f"[ERROR] Tavily 搜索失败: {e}")
            return {"documents": [Document(page_content=f"搜索失败: {e}")]}

        

    def router_question(self, state: GraphState) -> dict:
        print("🛜  问题路由选择")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("🌐 选择网络搜索")
            return {"type": "web_search"}
        print("📚 选择本地RAG")
        return {"type": "vectorestore"}

    def route_by_type(self, state: GraphState) -> str:
        return state["type"]
    
    def web_search(self, state: GraphState):
        print("---选择网络搜索---")
        return self.tavily_search(state)

    
    def vectorestore(self, state: GraphState):
        print("---选择RAG---")
        return self.rag_search(state)


def build_app():
    components = InflectionRAG()
    workflow = StateGraph(GraphState)

    workflow.add_edge(START, "route")
    workflow.add_node("route", components.router_question)
    
    workflow.add_conditional_edges(
        "route",
        components.route_by_type,
        {
            "web_search": "web_search",
            "vectorestore": "vectorestore",
        }
    )
    workflow.add_node("web_search", components.web_search)
    workflow.add_node("vectorestore", components.vectorestore)
    workflow.add_node("grade_documents", components.grade_documents)
    workflow.add_node("transform_query", components.transform_query)

    workflow.add_conditional_edges(
        "grade_documents",
        components.decide_to_generate,
        {
            "transform_query": "transform_query",
            "generation": "generation",
        },
    )

    workflow.add_edge("transform_query", "vectorestore")

    workflow.add_node("generation", components.generation)
    workflow.add_node("print", components.print)

    workflow.add_edge("web_search", "generation")
    workflow.add_edge("vectorestore", "grade_documents")
    workflow.add_edge("generation", "print")
    workflow.add_edge("print", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_app()
    app.invoke({"question": "LLM是什么？"})
    # print(result['documents'][0].page_content)










