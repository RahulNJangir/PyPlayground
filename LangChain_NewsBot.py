import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------
# 1. SET YOUR FREE GEMINI API KEY
# -------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual API key

# --------------------------------------------------------------
# 2. SETUP SEARCH TOOL & GEMINI MODEL
# --------------------------------------------------------------
# Free web search tool (No API key needed)
search_tool = DuckDuckGoSearchRun()

# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an executive news reporter. Read the raw search results and provide a clean, structured news briefing with an executive summary and key bullet points."),
    ("user", "Topic: {topic}\n\nSearch Results:\n{search_results}")
])

# Create the LangChain pipeline
chain = prompt | llm

# -------------------------------------------------------------
# 3. INTERACTIVE NEWS BOT
# -------------------------------------------------------------
def start_news_bot():
    print("=" * 60)
    print("🤖 Welcome to the Daily AI News Assistant!")
    print("Type 'exit' or 'quit' anytime to stop the program.")
    print("=" * 60 + "\n")

    while True:
        # Ask user for input
        topic = input("\n👉 Which topic do you want news updates on? ").strip()

        # Check if user wants to stop
        if topic.lower() in ["exit", "quit", "no", "stop"]:
            print("\n👋 Goodbye! Have a great day!")
            break

        if not topic:
            print("⚠️ Please enter a valid topic name.")
            continue

        print(f"\n🔍 Searching latest news for: '{topic}'...")

        try:
            # Step A: Fetch search results from DuckDuckGo
            raw_data = search_tool.invoke(f"latest news on {topic}")

            print("🤖 Processing news summary with Gemini AI...\n")

            # Step B: Pass search data to Gemini
            response = chain.invoke({
                "topic": topic,
                "search_results": raw_data
            })

            # Step C: Print formatted response
            print("=" * 60)
            print(f"📰 DAILY BRIEFING: {topic.upper()}")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

        except Exception as e:
            print(f"❌ Error fetching news: {e}")

if __name__ == "__main__":
    start_news_bot()