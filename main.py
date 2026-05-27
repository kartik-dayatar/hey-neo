from langchain_ollama import OllamaEmbeddings, ChatOllama

embeddings = OllamaEmbeddings(model="nmxbai-embed-large:latest")
llm = ChatOllama(model="gemma4:latest", temperature=0.2, system_prompt="You are profational therapist, you will help the user with their mental health issues. You will be kind and understanding, and you will provide helpful advice and support to the user.you will also ask the user questions to better understand their situation and provide more personalized advice. You will always be respectful and non-judgmental, and you will never criticize or belittle the user. Your goal is to help the user feel better and improve their mental health.")


for chunk in llm.stream("hey can you help me wiht my problem of not trusting others?"):
    print(chunk.content, end="", flush=True)
