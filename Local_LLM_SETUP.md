# 🧠 Local LLM Chatbot

Run **Llama 3.2** locally and chat with it from Python using **Ollama** and **LiteLLM**. Once the model is downloaded, prompts and responses are handled on your computer rather than sent to a hosted LLM API.

## 🔄 How it works

```text
You → Python chatbot → LiteLLM → Ollama → Llama 3.2
```

| Part | What it does |
| --- | --- |
| 🐍 Python | Runs the chat application |
| 🔌 LiteLLM | Connects Python to the model |
| 🦙 Ollama | Serves the model locally |
| 🤖 Llama 3.2 | Generates responses |

## 📋 Requirements

- Windows with Python and `pip`
- [Ollama for Windows](https://ollama.com/download)
- An internet connection for installation and the initial model download

Check that Python and pip are available:

```powershell
python --version
python -m pip --version
```

## ⚙️ Set up

### 1. 🦙 Install and start Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download), then open a new PowerShell window and check the installation:

```powershell
ollama --version
```

### 2. 🤖 Download Llama 3.2

```powershell
ollama pull llama3.2
```

Confirm that the model is installed:

```powershell
ollama list
```

### 3. 🔌 Install LiteLLM

From your project directory, run:

```powershell
python -m pip install litellm
```

### 4. 💬 Start chatting

Run the chatbot:

```powershell
python LiteLLM.py
```

Enter a message at the `You:` prompt. Type `exit` to end the chat.

## 🐍 What the Python app does

`LiteLLM.py` sends each prompt to the local Ollama service at `http://localhost:11434` and prints the model's response:

```python
from litellm import completion

response = completion(
    model="ollama/llama3.2",
    messages=[{"role": "user", "content": "What is machine learning?"}],
    api_base="http://localhost:11434",
)

print(response.choices[0].message.content)
```

The example above makes a single request. The included `LiteLLM.py` script wraps this call in a loop so you can continue chatting until you type `exit`.

## 🛠️ Troubleshooting

### ⚠️ Ollama is not responding

Make sure the Ollama application is running, then try:

```powershell
ollama run llama3.2
```

### 📥 The model is missing

Download it with `ollama pull llama3.2`, then run the chatbot again.

### 📦 Python cannot import LiteLLM

Install LiteLLM in the same Python environment used to run the script:

```powershell
python -m pip install litellm
```

## 📁 Project files

```text
.
├── LiteLLM.py
└── Local_LLM_SETUP.md
```

**✨ Stack:** 🐍 Python · 🔌 LiteLLM · 🦙 Ollama · 🤖 Llama 3.2
