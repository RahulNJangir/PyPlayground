import json, os
from getpass import getpass
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain_openrouter import ChatOpenRouter

MEMORY_FILE = r"C:\Users\rahul\OneDrive\Desktop\Python_pratice\chat_memory.json"

def load_memory(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except Exception as e:
            print(f"Error loading memory: {e}")
    return []

def save_memory(path, memory):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages_to_dict(memory), f, indent=2)
        print("Memory saved.")
    except Exception as e:
        print(f"Error saving memory: {e}")

api_key = getpass("Enter OpenRouter API key: ").strip()
if not api_key:
    raise ValueError("API key required.")

bot = ChatOpenRouter(
    model="google/gemini-2.5-flash", 
    api_key=api_key,
    max_tokens=1000
)

memory = load_memory(MEMORY_FILE)

print("Chatbot started. Type 'exit' to stop.")

try:
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        memory.append(HumanMessage(content=user_text))
        try:
            answer = bot.invoke(memory)
            print("Bot:", answer.content)
            memory.append(AIMessage(content=answer.content))
        except Exception as error:
            memory.pop()
            print(f"Error: {error}")
finally:
    save_memory(MEMORY_FILE, memory)
    print("Goodbye!")