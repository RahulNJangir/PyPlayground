from getpass import getpass  # accepts secret input without displaying

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def get_api_key(): #function to get the API key from user input, using getpass for secure input
    try:
        return getpass("Enter your Google API key: ").strip()
    except (EOFError, OSError):
        return input("Enter your Google API key: ").strip()


api_key = get_api_key()
if not api_key:
    raise ValueError("A Google API key is required to start the chatbot.")

bot = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
)

memory = []

print("Chatbot started. Type 'exit' to stop.")

try:
    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not user_text:
            print("Please enter a message.")
            continue

        memory.append(HumanMessage(content=user_text))
        print("------------")

        try:
            answer = bot.invoke(memory)
        except Exception as error:
            memory.pop()
            print(f"Request failed: {error}")
            continue

        print("Bot:", answer.content)
        memory.append(AIMessage(content=answer.content))
except (EOFError, KeyboardInterrupt):
    print("\nGoodbye!")
