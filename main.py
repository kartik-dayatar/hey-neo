from agents import build_agent
from tools import rewrite_query

from langchain.messages import HumanMessage, AIMessage, SystemMessage

print("What the problem dude neo is here......! \n")




print("="*100)
agent = build_agent()

chat_memory = []

while True:


    query = input(">>>")

    if query == "exit" or query == "Exit":
        break
    if query == "memory":
        print(chat_memory)
        for i in chat_memory:
            print(i.content)

    prompt = query
    
    chat_memory.append(HumanMessage(prompt))

    response = agent.invoke({"messages": chat_memory})
    response = response["messages"][-1]

    if isinstance(response.content, list):
        text = next((block["text"] for block in response.content if block.get("type") == "text"), "")
    else:
        text = response.content

    print(f"\n{text}\n")

    chat_memory.append(AIMessage(text))


print("Have a grate time...")