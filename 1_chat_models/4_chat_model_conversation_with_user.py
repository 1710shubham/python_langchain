from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage,HumanMessage,SystemMessage


#load environment variable from .env
load_dotenv()

#create a ChatOpenAI model
model = ChatOpenAI(model = "gpt-4o-mini")

chat_history = [] #Use a list to store message

#set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)#Add system message to chat history 

#chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) #Add user message

    #Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response)) #Add AI message

    print(f"AI:{response}")

print("----Message History----")
print(chat_history)