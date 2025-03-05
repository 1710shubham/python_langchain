from dotenv import load_dotenv
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_openai import ChatOpenAI
import uuid
import grpc



load_dotenv()
PROJECT_ID = "selflearn-ac07c"
SESSION_ID = f"user_session_{uuid.uuid4()}"
USER_ID = "user_1234"
COLLECTION_ID = f"user_{USER_ID}_chats"

# Initializing Firestore Client
print("Initializing Firestore Client...")
client = firestore.Client(project=PROJECT_ID)

# Initilizing Firestore Chat Message History
print("Initilizing Firestore Chat Message History...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_ID,
    client=client,
)
print("Chat History Initilized.")
print("Current Chat History:",chat_history.messages)

#  Initialized chat model
model = ChatOpenAI()
print("Start chatting with the AI. type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")