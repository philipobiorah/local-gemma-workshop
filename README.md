# Local Gemma Workshop

Build a general-purpose AI assistant with **Gemma 4**, **Ollama** and **Google Agent Development Kit (ADK)**, then extend it with a web-search tool.

Prepared by **Dr Philip Obiorah** for *Deploying LLMs Locally with Gemma*, part of **Sovereign AI: Building AI You Can Trust**.

## What is included?

| Example | Purpose | Data boundary |
| --- | --- | --- |
| `basic_agent` | A concise general help assistant | Model inference runs locally through Ollama |
| `basic_agent_with_search` | A general assistant that can retrieve Google search snippets | Gemma inference is local; search queries are sent to Serper |

The search example uses a **custom Python function**, `search_google`, that calls [Serper](https://serper.dev/), a third-party Google search-results service. It does **not** use ADK's Gemini-only built-in Google Search tool. No Gemini API key is required.

**Document Q&A / RAG is not included in this repository yet.** Neither example indexes uploaded documents.

## What you will learn

- Run Gemma 4 on your own computer.
- Connect ADK to Ollama through LiteLLM.
- Define an agent's model, instructions and purpose.
- Register a Python function as an agent tool.
- Inspect the difference between model responses and tool execution.
- Understand the boundary between local AI and online services.

## Why local AI matters

Local inference gives you control over the model runtime and where inference requests are processed. It can support offline use and reduce dependence on cloud inference providers.

Local deployment alone does not guarantee accurate answers, secure storage or complete AI sovereignty. The search-enabled example introduces an external dependency: its generated search queries leave your computer.

There is no cloud inference API fee for these local Gemma examples. Hardware, electricity and maintenance still have costs, and Serper has its own usage limits and pricing.

## Before you begin

Install:

1. **Python 3.11 or newer:** [python.org/downloads](https://www.python.org/downloads/). On Windows, select **Add Python to PATH**.
2. **Git:** [git-scm.com/downloads](https://git-scm.com/downloads/).
3. **Ollama:** [ollama.com/download](https://ollama.com/download).
4. A text editor and browser.

On Debian/Ubuntu, install Python, its virtual-environment support and Git through your package manager:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

Use the official Ollama instructions for your operating system. Launch the Ollama desktop application on Windows/macOS. On Linux, ensure the Ollama service is running; if needed, run `ollama serve` in a separate terminal.

Initial software, package and model downloads require internet access. Performance depends on your hardware; test your machine before a live demonstration.

## 1. Clone the repository

```bash
git clone https://github.com/philipobiorah/local-gemma-workshop.git
cd local-gemma-workshop
```

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Once activated, your terminal should show `(.venv)`.

If PowerShell prevents activation, you can use the environment's executables directly, such as `.\.venv\Scripts\python.exe` and `.\.venv\Scripts\adk.exe`.

## 3. Install dependencies

With the environment activated:

```bash
python -m pip install -r requirments.txt
adk --version
```

**The filename is currently spelled `requirments.txt` in this repository.** Use that exact spelling.

It installs Google ADK, Google Gen AI SDK and LiteLLM. Installing the Google SDK does not mean these agents use cloud Gemini: both agents select Ollama-hosted Gemma.

## 4. Download and check Gemma

With Ollama running:

```bash
ollama --version
ollama pull gemma4:e2b
ollama list
ollama run gemma4:e2b
```

Ask:

> Explain what an AI agent is using an everyday example.

Enter `/bye` to exit. Keep Ollama running for the ADK examples.

The default model is `gemma4:e2b`. Both agents use the `ollama_chat` LiteLLM provider and an 8,192-token context setting.

## 5. Configure the local runtime

Set these variables **before starting ADK**, so they are available when LiteLLM is imported.

### macOS / Linux

```bash
export OLLAMA_API_BASE="http://127.0.0.1:11434"
export GEMMA_MODEL="gemma4:e2b"
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export LITELLM_MODE="PRODUCTION"
export NO_PROXY="127.0.0.1,localhost"
```

### Windows PowerShell

```powershell
$env:OLLAMA_API_BASE="http://127.0.0.1:11434"
$env:GEMMA_MODEL="gemma4:e2b"
$env:LITELLM_LOCAL_MODEL_COST_MAP="True"
$env:LITELLM_MODE="PRODUCTION"
$env:NO_PROXY="127.0.0.1,localhost"
```

These settings select the local endpoint and use LiteLLM's packaged model-cost map instead of fetching it remotely. They do not constitute a firewall or a complete dependency-network audit.

Reapply the variables when opening a new terminal.

## 6. Run the basic assistant

From the repository root:

```bash
adk web --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** and select **basic_agent**.

Try:

- “Explain machine learning to a beginner.”
- “Help me write a polite meeting invitation.”
- “Suggest a four-week plan for learning Python.”

This agent has no registered tools. It answers using its model knowledge and conversation context, so it cannot verify current information through web search.

To use the terminal interface instead:

```bash
adk run basic_agent
```

Press **Ctrl+C** to stop the running interface.

## 7. Enable the search assistant

### Obtain a search key

Create an account at [Serper](https://serper.dev/) and obtain an API key. Check its current limits and pricing.

Set the key in the terminal where you will launch ADK:

macOS / Linux:

```bash
export SERPER_API_KEY="your_actual_key"
```

Windows PowerShell:

```powershell
$env:SERPER_API_KEY="your_actual_key"
```

For a reusable local configuration, ADK can load a `basic_agent_with_search/.env` file containing:

```dotenv
SERPER_API_KEY=your_actual_key
```

The repository's `.gitignore` excludes `.env`. Never commit credentials.

### Start and test

```bash
adk web --host 127.0.0.1 --port 8000
```

Select **basic_agent_with_search** and ask:

> Search Google for the official Google ADK documentation and give me its URL.

Inspect the ADK events to confirm that **search_google** was called. A response containing a URL does not, by itself, prove tool execution.

The tool returns up to five organic result titles, URLs and snippets. It does not download or read the full webpages.

If the key is missing, invalid or the service is unavailable, the function returns an error for the agent to handle. Tool use is requested through the agent's instructions; it is not guaranteed on every question.

## How the examples work

### Basic assistant

The ADK interface sends your question to the agent. LiteLLM connects to local Ollama, which runs Gemma and returns its response.

### Assistant with search

Gemma can request `search_google(query)`. ADK executes the Python function, which sends that query to Serper. The returned snippets become context for Gemma's answer.

Gemma inference remains local, but the search step requires internet. Do not describe the search-enabled example as fully offline.

## Reading the code

| File | Responsibility |
| --- | --- |
| `basic_agent/agent.py` | Local model configuration and basic assistant |
| `basic_agent/__init__.py` | Exposes the root agent |
| `basic_agent_with_search/agent.py` | Instructions and search-tool registration |
| `basic_agent_with_search/search_tools.py` | Serper request, result formatting and error handling |
| `basic_agent_with_search/__init__.py` | Exposes the search-enabled root agent |
| `requirments.txt` | Python dependencies |
| `.gitignore` | Excludes the environment, ADK state, Python caches and `.env` |

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `adk` command not found | Activate `.venv` and install dependencies |
| Ollama connection refused | Launch Ollama or check its service |
| Model not found | Run `ollama pull gemma4:e2b` |
| Agent missing from the selector | Run `adk web` from the repository root |
| Port 8000 is occupied | Stop the earlier server, or choose another port |
| `SERPER_API_KEY is not configured` | Set the key before launching ADK, then restart |
| Search returns HTTP 401/403 | Check the Serper key and account access |
| Search returns HTTP 429 | Check the provider's usage limits |
| Agent answers without searching | Explicitly request search and inspect the event trace |
| Slow responses | Allow time for loading and inference; close other demanding apps |

An ADK warning about `JSON_SCHEMA_FOR_FUNC_DECL` being experimental is not, by itself, an application failure. Look for subsequent errors if a request fails.

To try the larger E4B variant, download `gemma4:e4b`, change `GEMMA_MODEL` to that tag, and restart ADK. Check available memory first.

## Privacy and limitations

- Local inference does not make model answers reliable or current.
- Search queries are transmitted to Serper; snippets are external, untrusted content.
- The ADK development interface may retain local conversation/session data. Review its configuration before using sensitive information.
- Keep the development server bound to localhost; it is not an enterprise deployment.
- This repository does not include document upload, vector indexing or a Document Q&A application.
- Setup instructions are based on the source files; they are not a claim that live model behaviour has been validated on every operating system.

## Next learning step: local Document Q&A

A future extension can add local document extraction, EmbeddingGemma vectors and a SQLite index. Retrieved passages would provide evidence for Gemma's answers: **Retrieval-Augmented Generation (RAG)**.

That extension can keep document processing and inference local without requiring the web-search tool.

## References

- [Google ADK](https://adk.dev/)
- [ADK with Ollama](https://adk.dev/agents/models/ollama/)
- [Ollama documentation](https://docs.ollama.com/)
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Serper](https://serper.dev/)

Model weights and external services retain their own terms. This repository currently has no `LICENSE` file; do not infer a software license from the licenses of its dependencies.
