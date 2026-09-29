from agents import build_agent
from tools import rewrite_query
from ingest.ingest_chat import ingest_chat_turn

from langchain_core.messages import HumanMessage, AIMessage

print("\nWhat the problem dude neo is here......! \n")
print("="*100)
agent = build_agent()

chat_memory = []


def prep_chat_save() -> str:
    session = ""
    for msg in chat_memory:
        session += f"{msg.content}\n"
    return session


while True:

    query = input(">>>")

    if query.lower() in {"exit", "quit"}:
        break

    rewritten = rewrite_query(query)
    prompt = f"Question: {rewritten}"

    chat_memory.append(HumanMessage(prompt))

    response = agent.invoke({"messages": chat_memory})
    response = response["messages"][-1]

    if isinstance(response.content, list):
        text = next(
            (block["text"] for block in response.content if block.get("type") == "text"),
            "",
        )
    else:
        text = response.content

    print(f"\n{text}\n")

    chat_memory.append(AIMessage(text))

    # Persist this turn to the chat_history Qdrant collection
    ingest_chat_turn(user_input=query, neo_response=text)


print("Have a great time...")
save = input("Want to save chat? (y/n): ")
if save.strip().lower() == "y":
    print(prep_chat_save())