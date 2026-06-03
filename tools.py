from langchain_ollama import ChatOllama

import requests
from tavily import TavilyClient

from retrival import bm25_search, similarity_search, package_search

import os
from dotenv import load_dotenv


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
from langchain.tools import tool


def rewrite_query(query:str):

    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.2)
    system_prompt="""
    You are a query rewriting engine for a local AI assistant.
    Your goal is to transform the user's query into a clear, concise, and optimized version.
    The rewritten query should:
    - expain problem and mentain all relavent details
    - not contain any question marks
    - be suitable for a search engine
    - preserve the original intent and keywords
    - make sure to mentain user perspective not your's eg: he ask for "my" system so answer should be about user system not about my system
    - don't generate any extra lines just give the rewritten query
    - don't use "you" perspective use "my" perspective
    - if input is in different language then convert it to english and give responce in english
    
    Input Query: '{query}'.
    """

    prompt=system_prompt.format(query=query)
    try:
        response= llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(e)
        return query

@tool
def web_search_tool(query:str):
    """
    this tool will be used to search the web for missing/latest information if it is not present in context.
    """

    content = []
    client = TavilyClient(TAVILY_API_KEY)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=3,
        include_images=False
    )

    for result in response['results']:
        content.append(f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}")

    return "\n\n".join(content)
     

@tool
def bm25_search_tool(query:str):
    """
    this tools is used to retrive any information regarding any perticular package,library or path 
    use this tools if needed to retrive or check any specific detail from local context.
    this tool should only be used on exact package/path/library search query not for general search query
    """

    return bm25_search(query)

@tool
def similarity_search_tool(query:str):
    """
    this tool is used to retrive information regarding system information
    use this tool if needed to retrive any general information from local context.
    """
    return similarity_search(query)

@tool
def package_search_tool(query:str):
    """
    this tool is only and only be used to retrive information regarding installed packages
    """
    return package_search(query)