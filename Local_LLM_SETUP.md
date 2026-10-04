# 🧠 Local LLM Platform

<p align="center">
  <strong>🐍 Python · 🔌 LiteLLM · 🦙 Ollama · 🤖 Llama 3.2</strong>
</p>

<p align="center">
  A simple local LLM stack for running and interacting with Llama 3.2 through Python.
</p>

---

## 🎯 Overview

This project demonstrates how to connect a **Python application** to a locally running **Llama 3.2** model using **Ollama** and **LiteLLM**.

> 💡 **Goal:** Build a simple, local, and easy-to-understand LLM application without relying on a cloud API.

---

## 🏗️ Architecture

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

## 🧩 Components

| Component | Role |
|---|---|
| **Python** | Application layer |
| **LiteLLM** | LLM interface / gateway |
| **Ollama** | Local model runtime |
| **Llama 3.2** | Language model |

## ⚙️ Prerequisites

Check Python and pip:

```powershell
python --version
pip --version
```

## 📦 Installation

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

## 🐍 Python Integration

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

## 🔄 Request Flow

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

## 📚 Key Concepts

- **Python** — builds the application.
- **LiteLLM** — provides a common interface for calling LLMs.
- **Ollama** — runs LLMs locally.
- **Llama 3.2** — generates the responses.

---

## 🚀 Quick Reference

| Component | Purpose |
|---|---|
| 🐍 **Python** | Application layer |
| 🔌 **LiteLLM** | Unified LLM interface |
| 🦙 **Ollama** | Local model runtime |
| 🧠 **Llama 3.2** | Language model |

### 🔄 Request Flow

```text
👤 User
   ↓
🐍 Python Application
   ↓
🔌 LiteLLM
   ↓
🦙 Ollama
   ↓
🧠 Llama 3.2
   ↓
💬 Response
```

---

<div align="center">

### ⭐ Local LLM Stack

**Python + LiteLLM + Ollama + Llama 3.2**

*Run and interact with an LLM locally through a simple Python application.*

</div>


---

## ✨ Project Highlights

| ✨ Feature | 📝 Description |
|---|---|
| 🏠 **Local AI** | Run the LLM directly on your machine |
| 🔌 **Unified Interface** | Use LiteLLM to communicate with the model |
| 🦙 **Ollama Runtime** | Manage and serve the local model |
| 🤖 **Llama 3.2** | Generate natural-language responses |
| 🐍 **Python Ready** | Easily integrate the stack into Python applications |
| 🔒 **Local Processing** | No external LLM API is required for inference |

---

## 🧭 Architecture at a Glance

```text
        👤 User
           │
           ▼
    🐍 Python Application
           │
           ▼
       🔌 LiteLLM
           │
           ▼
       🦙 Ollama
           │
           ▼
      🤖 Llama 3.2
           │
           ▼
      💬 AI Response
```

---

<div align="center">

## 🚀 Local LLM • Simple • Private • Extensible

**🐍 Python + 🔌 LiteLLM + 🦙 Ollama + 🤖 Llama 3.2**

⭐ *A practical foundation for building local AI applications.*

</div>
