# Local LLM Platform

A lightweight local LLM application stack using **Python, LiteLLM, Ollama, and Llama 3.2**.

## Architecture

```text
User
  │
  ▼
Python Application
  │
  ▼
LiteLLM
  │
  ▼
Ollama
  │
  ▼
Llama 3.2
  │
  ▼
AI Response
```

## Components

| Component | Role |
|---|---|
| **Python** | Application layer |
| **LiteLLM** | LLM interface / gateway |
| **Ollama** | Local model runtime |
| **Llama 3.2** | Language model |

## Prerequisites

Check Python and pip:

```powershell
python --version
pip --version
```

## Installation

### 1. Install Ollama

```powershell
winget install --id Ollama.Ollama -e
```

Verify:

```powershell
ollama --version
```

### 2. Install Llama 3.2

```powershell
ollama pull llama3.2
```

Verify the model:

```powershell
ollama list
```

Run a test:

```powershell
ollama run llama3.2 "What is AWS?"
```

### 3. Install LiteLLM

```powershell
pip install litellm
```

Verify:

```powershell
python -c "import litellm; print(litellm.__version__)"
```

## Python Integration

Create `LiteLLM.py`:

```python
from litellm import completion

response = completion(
    model="ollama/llama3.2",
    messages=[
        {"role": "user", "content": "What is AWS?"}
    ],
    api_base="http://localhost:11434"
)

print(response.choices[0].message.content)
```

Run:

```powershell
python LiteLLM.py
```

## Request Flow

```text
Python
  ↓
LiteLLM
  ↓
Ollama
  ↓
Llama 3.2
  ↓
Response
```

## Key Concepts

- **Python** — builds the application.
- **LiteLLM** — provides a common interface for calling LLMs.
- **Ollama** — runs LLMs locally.
- **Llama 3.2** — generates the responses.

## Project Structure

```text
local-llm-platform/
├── README.md
├── LiteLLM.py
└── requirements.txt
```

`requirements.txt`:

```text
litellm
```

## Stack Summary

```text
Python → LiteLLM → Ollama → Llama 3.2
```

This setup allows a Python application to use **Llama 3.2 locally through Ollama with LiteLLM as the interface**.
