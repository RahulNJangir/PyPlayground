# 🧠 Local LLM Chatbot

Run **Llama 3.2** on your computer and chat with it from Python using **Ollama** and **LiteLLM**. After the model is downloaded, inference runs locally without sending prompts to a hosted LLM API.

## 🏗️ How it works

```text
👤 You → 🐍 Python app → 🔌 LiteLLM → 🦙 Ollama → 🤖 Llama 3.2
```

| Component | Role |
| --- | --- |
| 🐍 **Python** | Runs the chatbot |
| 🔌 **LiteLLM** | Connects the app to the model |
| 🦙 **Ollama** | Serves the model locally |
| 🤖 **Llama 3.2** | Generates responses |

## 📋 Prerequisites

- Windows with Python and `pip` installed
- Ollama for Windows
- Internet access to install the software and download the model

Check Python and pip from PowerShell:

```powershell
python --version
python -m pip --version
```

## ⚙️ Setup

### 1. Install Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download), or use `winget`:

```powershell
winget install --id Ollama.Ollama -e
```

Open a new PowerShell window and verify the installation:

```powershell
ollama --version
```

### 2. Download Llama 3.2

```powershell
ollama pull llama3.2
```

Check that the model is installed:

```powershell
ollama list
```

### 3. Install LiteLLM

From the project directory, install LiteLLM into the Python environment used to run the chatbot:

```powershell
python -m pip install litellm
```

### 4. Run the chatbot

From the directory containing `LiteLLM.py`, start the app:

```powershell
python LiteLLM.py
```

Type a message at the `You:` prompt. Enter `exit` to quit.

## 💬 What the app does

`LiteLLM.py` sends each message to the local Ollama service at `http://localhost:11434`, using the `ollama/llama3.2` model, and displays the response. It keeps prompting until you enter `exit`.

## 🛠️ Troubleshooting

### Ollama is not responding

Make sure Ollama is installed and running. You can also start the model manually:

```powershell
ollama run llama3.2
```

### The model is not installed

Download it, then try again:

```powershell
ollama pull llama3.2
```

### Python cannot import LiteLLM

Install LiteLLM with the same Python command you use to run the app:

```powershell
python -m pip install litellm
```

## 📁 Project files

```text
.
├── LiteLLM.py
└── Local_LLM_SETUP.md
```

---

**✨ Local AI stack:** Python · LiteLLM · Ollama · Llama 3.2

## 🛑 Stop the model

To stop the running Llama 3.2 model in PowerShell:

```powershell
ollama stop llama3.2
```
