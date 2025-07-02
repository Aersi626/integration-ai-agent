# Apache Camel YAML Transformation with LangGraph and Langfuse Observability

This project demonstrates an Apache Camel YAML transformation pipeline using LangChain, LangGraph, and Langfuse for observability. It modifies Camel YAML configurations based on natural language instructions.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [How to Run](#how-to-run)
- [Langfuse Observability](#langfuse-observability)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Features

* **YAML Transformation:** Modifies Apache Camel YAML routes.
* **RAG Pipeline:** Uses LangChain for document loading, splitting, embedding, and retrieval.
* **LangGraph Workflow:** Orchestrates retrieval and generation.
* **Langfuse Observability:** Traces LLM application execution for debugging and monitoring.
* **Apache Camel Karavan Integration:** Provides a sample Camel Karavan YAML route for VS Code.

## Prerequisites

Ensure these are installed:

* **Docker Desktop:** For Langfuse and Ollama.
    * [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
* **Git:** For cloning.
    * [Download Git](https://git-scm.com/downloads)
* **Python 3.9+:** For the script.
    * [Download Python](https://www.python.org/downloads/)
* **VS Code:** (Recommended for Karavan).
    * [Download VS Code](https://code.visualstudio.com/)
* **Apache Camel Karavan VS Code Extension:**
    * Install from VS Code Extensions Marketplace.
* **Camel JBang:** (Required by Karavan).
    * Install via `curl -Ls https://sh.jbang.dev | bash -s - app install camel@apache/camel` (Linux/macOS) or follow [JBang guide](https://camel.apache.org/manual/camel-jbang.html) for Windows.

## Setup

### Docker Compose for Langfuse & Ollama

1.  **Clone Langfuse repo:**
    ```bash
    git clone [https://github.com/langfuse/langfuse.git](https://github.com/langfuse/langfuse.git)
    cd langfuse
    ```
2.  **Start Langfuse services:**
    ```bash
    docker compose down -v # Clears old data
    docker compose up -d   # Starts Langfuse
    ```
    Wait for services to be healthy (`docker compose ps`).
3.  **Access Langfuse UI (`http://localhost:3000`):** Create Organization, Admin User, and a Project. **Copy new `Public Key` and `Secret Key`** from Project Settings.
4.  **Start Ollama:**
    * [Download ollama.ai](https://ollama.ai/)
    * Pull models: `ollama pull nomic-embed-text`, `ollama pull codellama:7b`

### Python Environment

1.  **Create project directory.**
2.  **Create & activate virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install langchain-community langchain-text-splitters langchain-core langgraph langfuse tqdm
    ```
4.  **Create `data/DSL_example.txt`** in project root with relevant content.
5.  **Save `rag_pipeline_with_langfuse.py`** in project root.
    * **Update `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`** in the script.
    * Ensure `os.environ["LANGFUSE_HOST"] = "http://localhost:3000"`.

### Camel Karavan Setup in VS Code

1.  **Open VS Code.**
2.  **Install Apache Camel Karavan extension.**
3.  **Create new Karavan application:**
    * Command Palette (`Ctrl+Shift+P`) > `Karavan: Create Application`.
    * Choose `Camel JBang` (or preferred runtime).
    * Name it (e.g., `my-camel-karavan-project`).
4.  **Place `timer-hello-world.yaml`** inside `my-camel-karavan-project/src/main/resources/camel/`.
    ```yaml
    - route:
        id: timer-hello-world-route
        from:
          uri: "timer:myTimer?period=2000&repeatCount=5"
          steps:
            - log:
                message: "hello world"
    ```
5.  **Open the Karavan project folder** in VS Code (`File` > `Open Folder...`).

## How to Run

### Run the Python Transformation Script

1.  **Activate virtual environment.**
2.  **Run from terminal:**
    ```bash
    python rag_pipeline_with_langfuse.py
    ```
    Observe progress and final YAML output.

### Run the Camel Karavan Route

1.  **In VS Code, open `timer-hello-world.yaml`.**
2.  **Click "Run" (play icon)** in the top-right editor corner.
    "hello world" will print in VS Code terminal every 2 seconds, 5 times.

## Langfuse Observability

After running Python script:

1.  Go to `http://localhost:3000`.
2.  Navigate to **"Traces"**.
3.  Find and click your `rag_pipeline_with_langfuse.py` trace for details.

## Project Structure


.
├── data/
│   └── DSL_example.txt
├── rag_pipeline_with_langfuse.py
├── docker-compose.yml            # (From Langfuse repo)
└── my-camel-karavan-project/
├── src/
│   └── main/
│       └── resources/
│           └── camel/
│               └── timer-hello-world.yaml
└── ...


## Troubleshooting

* **`git : The term 'git' is not recognized...`**: Install Git or ensure it's in PATH. Restart terminal.
* **`HTTPConnectionPool(host='localhost', port=7000): Max retries exceeded...`**: Python script is connecting to wrong Langfuse port. Verify `LANGFUSE_HOST` is `http://localhost:3000` in `rag_pipeline_with_langfuse.py` and run the saved version.
* **`No runtime configured! Create application!` (in Karavan)**: Create Karavan application, place YAML in `src/main/resources/camel/`, open project folder in VS Code, ensure Camel JBang is installed.
* **Traces not appearing in Langfuse UI**: Check Docker container health, API keys, `LANGFUSE_HOST`, `langfuse_handler.client.flush()`, and Python script logs.
