# Local Gemma Workshop

Agents built with [Google ADK](https://google.github.io/adk-docs/) that run on local Gemma models through [Ollama](https://ollama.com). No cloud API keys.

| Folder | What it is |
| --- | --- |
| `basic_agent/` | Workshop example agent |
| `basic_agent_with_search/` | Workshop example agent with search |
| `document_qa/` | **Document Q&A**: ask questions about your own files and get answers with verified quotations (documented below) |

---

## Document Q&A

Add PDF, Word, text or Markdown files, ask a question, and get an answer backed by quotations that are checked against the source text. Everything runs on your computer: the documents, the search index and both models.

There are two ways to use the same agent:

```bash
python app.py     # custom web page at http://127.0.0.1:8765
adk web           # ADK's built-in chat window; pick "document_qa" in the dropdown
```

Both read the same `document_qa/documents/` folder and the same index, so a file added in one shows up in the other.

### How it works

```
document_qa/documents/   your files (the single source of truth)
        │  every question starts by syncing this folder
        ▼
split into ~1000-character chunks → embedded with embeddinggemma → saved in SQLite
        │
question ──► search in code (top 5 chunks + the opening chunk of each document)
        │
        ▼
Gemma (via ADK) reads the question and passages, replies with JSON: answer + quotations
        │
        ▼
every quotation is checked: it must be an exact piece of a retrieved passage
        │
        ▼
answer shown with only the verified quotations
```

Two design choices are worth knowing:

- **Search happens in code, not by the model.** Small local models often skip optional tool calls, so the first search is forced and the passages are placed in the prompt. The model keeps one optional extra search tool for when the passages are not enough.
- **Unverified answers are never shown.** If none of the model's quotations can be found in the retrieved passages, you get a notice instead of the answer.

### Requirements

- Python 3.10 or newer (tested on 3.13)
- [Ollama](https://ollama.com) installed and running
- Enough memory for a small Gemma model (`gemma4:e2b`) plus the embedding model

### Setup

From the workshop folder:

```bash
# 1. Create and activate a virtual environment (skip if .venv already exists)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install the Python packages into that environment
python -m pip install -r requirments.txt

# 3. Download the two models (one-time, needs internet)
ollama pull gemma4:e2b
ollama pull embeddinggemma:300m
```

`requirments.txt` should contain:

```
google-adk==2.11.0
litellm==1.105.0
pypdf==6.10.0
python-docx==1.2.0
```

Keep the virtual environment active whenever you run `python app.py` or `adk web`, so both use the same packages.

### Configuration

Settings go in the `.env` file in the workshop folder. Both `python app.py` and `adk web` read it.

| Variable | Default | Allowed values |
| --- | --- | --- |
| `GEMMA_MODEL` | `gemma4:e2b` | `gemma4:e2b`, `gemma4:e4b` (larger, better at following instructions; run `ollama pull gemma4:e4b` first) |

### Adding and removing documents

Supported files: **PDF, DOCX, TXT, Markdown**.

| Action | Custom page (`python app.py`) | `adk web` |
| --- | --- | --- |
| Add | Choose files → **Upload and index** | Attach a file in the chat message |
| Remove | **Remove** button | Delete the file from the folder |

In every case the file lives in `document_qa/documents/`. You can also copy files into that folder by hand; the next question or page refresh picks them up. Deleting a file removes it and its chunks from the index.

Limits: 10 MiB per file, 200 PDF pages, 500,000 characters of text per document, 2,000 chunks in total. Scanned PDFs have no text layer and are rejected, because there is no OCR step. Identical files count once.

### Reading an answer

- **Verified quotations** are exact excerpts from your documents, with the file name and page.
- *"Some generated citations failed verification"* means the model also produced a quotation that is not in the passages. Treat the answer with caution and check the sources.
- *"I could not produce an answer with verified source quotations"* means none of its quotations checked out. Rephrase the question, or try `gemma4:e4b`.
- *"I could not find that in the uploaded documents"* is a genuine "not found".

Questions that name content ("What does the paper say about debugging?") work better than vague ones. Questions about the title or author work too, because the opening chunk of each document is always included.

### Project layout

```
local-gemma-workshop/
├── app.py                   web server for the custom page (python app.py)
├── static/                  the custom page: index.html, app.js, style.css
├── document_qa/             the agent; adk web finds it here
│   ├── agent.py             root_agent, system prompt and model settings
│   ├── callbacks.py         before/after each model call: attachments, folder sync, search, quote check
│   ├── tools.py             the optional extra search tool
│   ├── store.py             documents folder → chunks → embeddings → SQLite, and search
│   ├── grounding.py         parsing the model's reply and verifying quotations
│   ├── documents/           put your files here
│   └── data/                the search index (created automatically)
├── tests/                   offline tests
├── .env                     settings
└── requirments.txt          Python packages
```

`app.py` contains no agent logic. It imports the agent from `document_qa`, so both entry points behave the same.

### Tests

The tests use a fake embedder and a fake chat model, so they need neither Ollama nor the models:

```bash
python -m pip install pytest
python -m pytest tests -q
```

### Privacy and safety

- Documents, vectors and answers stay on your computer. `python app.py` listens on `127.0.0.1` only and rejects requests from other hosts or without its custom header.
- Document text is treated as untrusted. The prompt tells the model to ignore instructions inside documents, and the quotation check means a document cannot make the app show text that is not in the documents.
- Small models can still be wrong. Always check the quotations.
- Add these lines to `.gitignore` to keep private documents and the index out of git:

  ```
  document_qa/data/
  document_qa/documents/*
  !document_qa/documents/.gitkeep
  .adk/
  ```

### Troubleshooting

| Message or symptom | Fix |
| --- | --- |
| `No module named 'pypdf'` (or `docx`) | `python -m pip install pypdf python-docx` in the active environment |
| `Ollama request failed` | Start Ollama (open the app or run `ollama serve`) and check both models are pulled: `ollama list` |
| "Models missing" under the page title | `ollama pull gemma4:e2b` and `ollama pull embeddinggemma:300m`, then refresh |
| "No documents are indexed yet" | Add a file (see above) |
| First question is slow | The model is loading into memory; later questions are faster while it stays loaded (5 minutes) |
| `Embedding dimensions changed` | The embedding model changed; delete `document_qa/data/index.sqlite3` and ask again to re-index |
| `adk web` shows the raw JSON answer first | Turn off token streaming in the `adk web` settings |
| `adk web` does not list `document_qa` | Run it from the workshop folder (the one that contains `document_qa/`), not from inside a subfolder |
