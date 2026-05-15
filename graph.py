import os
from RAGtool import RAGTOOL
from router import bulid_question_router
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
    ragtool: RAGTOOL
    question: str
    generation: str
    documents: list[Document]

class InflectionRAG:
    def __init__(self) -> None:
        self.question_router = bulid_question_router()
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def tavily_search(self, state: GraphState) -> dict:
        try:
            print(f"网络搜索: {state['question']}")
            response = self.tavily_client.search(
                query=state["question"],
                search_depth="basic",
                max_results=2,
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

        

    def router_question(self, state: GraphState) -> str:
        print("🛜 问题路由选择")
        question = state["question"]
        source = self.question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("🌐 选择网络搜索")
            state["type"] = "web_search"
            return "web_search"
        print("📚 选择本地RAG")
        state["type"] = "vectorestore"
        return "vectorestore"
    
    def web_search(self, state: GraphState):
        print("---选择网络搜索---")
        return self.tavily_search(state)

    
    def vectorestore(self, state: GraphState):
        print("选择RAG")

def build_app():
    components = InflectionRAG()
    workflow = StateGraph(GraphState)

    workflow.add_conditional_edges(
        START,
        components.router_question,
        {
            "web_search": "web_search",
            "vectorestore": "vectorestore",
        }
    )
    workflow.add_node("web_search", components.web_search)
    workflow.add_node("vectorestore", components.vectorestore)
    workflow.add_edge("web_search", END)
    workflow.add_edge("vectorestore", END)

    return workflow.compile()


if __name__ == "__main__":
    rag_tool = RAGTOOL(file_path="./pdf_file/Happy-LLM-0727.pdf")
    app = build_app()
    result = app.invoke({"question": "我想买华为最新款的手机给我推荐", "ragtool": rag_tool})
    print(result)










