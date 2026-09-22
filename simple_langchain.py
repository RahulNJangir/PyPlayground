"""A very small LangChain example.

Install the dependency first:
    python -m pip install langchain-openai

Optionally set OPENAI_API_KEY before running. If it is not set, the program
asks for the key without displaying it on screen.
"""

import os
from getpass import getpass

from langchain_openai import ChatOpenAI


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
    if not api_key.strip():
        raise ValueError("An OpenAI API key is required.")

    question = input("Ask a question: ").strip()
    if not question:
        print("Please enter a question.")
        return

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key,
    )
    response = model.invoke(question)
    print("\nAnswer:")
    print(response.content)


if __name__ == "__main__":
    main()
