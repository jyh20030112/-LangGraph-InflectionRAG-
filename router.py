from typing import Literal
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import os

load_dotenv()


class RouterQuery(BaseModel):
    datasource: Literal['vectore', 'web_search'] = Field(
        ...,
        description='Given a user question choose to route it to web search or a vectorstore.'
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

def build_question_rewriter():
    
    system = (
    "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短。"
)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "这个是原始问题:\n\n{question}\n\n,以上问题改写为一个更清晰、更精准的优化版本。给出3个不同表述的查询，用逗号分开。",
            ),
        ]
    )

    llm = ChatTongyi(
        dashscope_api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
    )

    return prompt | llm | StrOutputParser()

def build_retrieval_grader():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个评分员，负责评估检索到的文档与用户问题的相关性。
                    如果文档包含与用户问题相关的关键词或语义含义，则将其评为相关。
                    这不需要是一个严格的测试。目标是过滤掉错误的检索结果。
                    给出一个评分 "yes" 或 "no"，以表明该文档是否相关。"""),
        ("human", "检索文档:\n\n{document}\n\n用户问题: {question}"),
    ])

    llm = ChatTongyi(
        dashscope_api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
    )

    structured_llm = llm.with_structured_output(GradeDocuments)

    return prompt | structured_llm


def build_model_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手。由于检索系统未能找到相关信息，请你直接依靠自身的知识储备来回答用户的问题。

            请遵循以下原则：
            1. 如实告知：回答前先简要说明"未能在知识库中找到相关资料，以下回答基于通用知识"
            2. 知识准确：给出有事实依据的回答，引用可靠的数据、理论和概念，避免凭空猜测
            3. 结构清晰：使用分点、分段等方式组织回答，让内容易于理解和阅读
            4. 边界意识：对于不确定或超出知识范围的内容，明确说明"此部分信息可能需要进一步查证"
            5. 实用导向：优先给出对用户有实际帮助的信息，例如原理、方法、建议或进一步探索的方向
            6. 简洁有力：避免冗长的铺垫，直击问题核心，用简洁的语言表达深刻的观点"""),
        ("human", "{question}"),
    ])

    llm = ChatTongyi(
        dashscope_api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
    )

    return prompt | llm | StrOutputParser()


def build_rag_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的RAG问答助手。你的任务是根据提供的上下文信息，准确、清晰地回答用户的问题。

            请遵循以下原则：
            1. 基于提供的上下文内容回答问题，不要添加上下文以外的信息
            2. 如果上下文中没有足够的信息来回答问题，请明确说明"根据提供的资料，无法找到相关信息"
            3. 保持回答简洁明了，直接回应用户的问题
            4. 如果上下文中有相关的引用来源，可以适当提及

            上下文信息：
            {context}"""),
                    ("human", "{question}")
    ])
    
    llm = ChatTongyi(
        dashscope_api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
    )
    
    return prompt | llm | StrOutputParser()

def bulid_question_router():
    '''
    问题选择路由: 选择RAG还是网络
    '''
    llm=ChatTongyi(
        dashscope_api_key=os.getenv("API_KEY"),
        model=os.getenv("CHAT_MODEL"),
    )

    structured_llm_router = llm.with_structured_output(RouterQuery)

    data = '关于LLM大语言模型的知识'

    system = f"""你是一个问题路由助手。你的任务是根据用户的问题内容，判断应该使用向量数据库检索（vectorstore）还是网络搜索（web_search）来回答该问题。
                向量数据库现在包含{data}

                判断规则：
                1. 如果问题涉及PDF文件中的具体内容、文档知识、历史记录、已存储的资料等，选择 'vectorstore'
                2. 如果问题涉及实时信息、最新新闻、当前事件、天气、股价、需要联网获取的信息等，选择 'web_search'

                请仔细分析用户问题，选择最合适的数据源。"""

    router_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ('human',"{question}")
        ]
    )

    return router_prompt | structured_llm_router

if __name__ == '__main__':
    source = bulid_question_router().invoke({"question": '我想学习LLM'})
    print(source.datasource)


