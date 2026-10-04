from litellm import completion

print("=" * 50)
print("        Llama 3.2 LiteLLM Chatbot")
print("        Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        response = completion(
            model="ollama/llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": user_input
                }
            ],
            api_base="http://localhost:11434"
        )

        answer = response.choices[0].message.content
        print("\nLlama: " + answer)

    except Exception as e:
        print("\nError:", e)