from getpass import getpass

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def get_api_key():
    try:
        return getpass("Enter your Google API key: ")
    except (EOFError, OSError):
        return input("Enter your Google API key: ").strip()


api_key = get_api_key()

bot = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)

memory = []

print("Chatbot started. Type 'exit' to stop.")

while True:
    user_text = input("You: ").strip()

    if user_text.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    memory.append(HumanMessage(content=user_text))

    answer = bot.invoke(memory)
    print("Bot:", answer.content)

    memory.append(AIMessage(content=answer.content))
