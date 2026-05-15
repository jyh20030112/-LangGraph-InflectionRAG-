from typing import Literal
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
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


