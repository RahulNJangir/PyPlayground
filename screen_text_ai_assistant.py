import os
import pyautogui
import pytesseract
from PIL import Image
from openai import OpenAI

# ==============================
# CONFIGURATION
# ==============================

# Put your API key in an environment variable:
# Windows PowerShell:
# $env:OPENAI_API_KEY="your_api_key"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Change this if Tesseract is installed somewhere else
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==============================
# TAKE SCREENSHOT
# ==============================

def capture_screen():
    print("\n📸 Capturing screen...")

    screenshot = pyautogui.screenshot()
    screenshot.save("screen_capture.png")

    return screenshot


# ==============================
# OCR - READ SCREEN
# ==============================

def extract_text(image):
    print("🔎 Reading text from screen...")

    text = pytesseract.image_to_string(image)

    return text.strip()


# ==============================
# ASK AI
# ==============================

def ask_ai(text):

    if not text:
        return "No readable text was detected on the screen."

    print("🤖 Asking AI...")

    response = client.responses.create(
        model="gpt-5-mini",
        input=f"""
You are a helpful question-answering assistant.

The following text was detected from a computer screen:

{text}

Identify the main question or task.

Give the correct answer in a simple and clear way.
If it is a programming or technical question, explain the answer briefly.
If there is no clear question, summarize what the screen contains.
"""
    )

    return response.output_text


# ==============================
# MAIN PROGRAM
# ==============================

def main():

    print("=" * 50)
    print("        SCREEN AI QUESTION ANSWERER")
    print("=" * 50)

    input("\nPress ENTER to scan your screen...")

    # Screenshot
    image = capture_screen()

    # OCR
    text = extract_text(image)

    print("\n" + "=" * 50)
    print("TEXT DETECTED:")
    print("=" * 50)

    print(text)

    # AI
    answer = ask_ai(text)

    print("\n" + "=" * 50)
    print("AI ANSWER:")
    print("=" * 50)

    print(answer)

    print("\nScreenshot saved as: screen_capture.png")


if __name__ == "__main__":
    main()