from agents import answering_agent
from tools import rewrite_query

query = input("What the problem dude neo is here......! \n")


prompt = rewrite_query(query)
prompt = f"Question: {prompt}"

print("="*100)
print(answering_agent(prompt))
