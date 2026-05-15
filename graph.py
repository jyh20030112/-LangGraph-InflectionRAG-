from RAGtool import RAGTOOL
from router import bulid_question_router
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph

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

    def router_question(self, state: GraphState) -> str:
        print("🛜问题路由选择")
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
        print("选择网络")
    
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
    result = app.invoke({"question": "我想学习LLM", "ragtool": rag_tool})
    print(result)










