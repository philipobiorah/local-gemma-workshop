"""Browser UI for the document Q&A agent:  python app.py  ->  http://127.0.0.1:8765

This file is only the web server. The agent, the document index and the answer
checking all live in the document_qa package, which `adk web` uses too, so both
front ends share one agent and one documents folder:

  Browser UI -> Handler (this file) -> document_qa.root_agent (ADK Runner)
                                    -> document_qa.store (documents/ folder + SQLite index)
"""
# --- Imports and settings ----------------------------------------------------
import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Load the workshop's .env (the file adk web reads) before importing the agent,
# because the agent reads GEMMA_MODEL when it is imported.
try:
    from dotenv import load_dotenv  # installed with google-adk
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from document_qa.agent import ANSWER_MODEL, root_agent
from document_qa.store import EMBED_MODEL, get_store

APP_NAME = 'document_qa'  # must match the agent folder's name
PORT = 8765
ASK_LOCK = threading.Lock()  # one question at a time: the local model is slow and memory-hungry
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'application/javascript'),
          '/style.css': ('style.css', 'text/css')}  # fixed allow-list of files we serve


# --- Talking to the agent ----------------------------------------------------
async def ask_agent(question):
    """Run the shared agent once and return its verified answer as a dict for the UI."""
    # A fresh in-memory session per question, so earlier answers never leak in.
    sessions = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=sessions)
    session = await sessions.create_session(app_name=APP_NAME, user_id='browser')
    message = types.Content(role='user', parts=[types.Part(text=question)])
    reply, data = '', None
    async for event in runner.run_async(user_id='browser', session_id=session.id, new_message=message):
        if event.is_final_response() and event.content:
            reply = ''.join(p.text or '' for p in event.content.parts or [])
            data = (event.custom_metadata or {}).get('document_qa')  # set by the agent's after-model callback
    if not data:
        # No structured answer means the agent explained a problem instead
        # (for example "No documents are indexed yet."): show it as an error.
        raise ValueError(reply or 'The agent produced no answer. Retry with a shorter question.')
    return {**data, 'model': ANSWER_MODEL}


def ask(question):
    if not isinstance(question, str) or not question.strip() or len(question) > 1000:
        raise ValueError('Enter a question of 1 to 1,000 characters.')
    with ASK_LOCK:
        return asyncio.run(ask_agent(question))  # handler threads are synchronous; ADK is async


# --- Document actions (all go through the documents folder) -----------------
def upload(name, content):
    """Save an uploaded file into the documents folder and index it."""
    store = get_store()
    saved = store.add_file(name, content)  # raises ValueError for unsupported or oversized files
    problems = store.sync()
    indexed = {d['name']: d for d in store.documents()}
    if saved in indexed:
        return {'name': saved, 'chunks': indexed[saved]['chunks']}
    # Not indexed: remove the file so it does not linger, and explain why.
    store.remove_file(saved)
    reason = next((p.split(': ', 1)[1] for p in problems if p.startswith(saved + ':')), None)
    raise ValueError(reason or 'This document is already indexed.')


def delete(ident):
    """Delete a document's file; the sync forgets it and its chunks."""
    store = get_store()
    for doc in store.documents():
        if doc['id'] == ident:
            store.remove_file(doc['name'])
    store.sync()


def documents():
    """List documents, first picking up files added to or removed from the folder by hand."""
    store = get_store()
    store.sync()  # problems are ignored here; they surface when a question is asked
    return store.documents()


def status():
    """Report whether both Ollama models are installed."""
    names = get_store().embedder.installed()
    return {'model': ANSWER_MODEL, 'embedding_model': EMBED_MODEL,
            'ready': all(m in names for m in (ANSWER_MODEL, EMBED_MODEL)), 'installed': names}


# --- HTTP server -------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    """Serves the web page and the JSON API on localhost only."""

    def log_message(self, *args):
        pass  # Avoid recording questions and filenames in access logs.

    def send(self, code, payload, mime='application/json'):
        """Send a response: JSON for dicts/lists, raw bytes for static files."""
        data = json.dumps(payload).encode() if mime == 'application/json' else payload
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')          # never cache documents/answers
        self.send_header('X-Content-Type-Options', 'nosniff')  # browser must trust our content type
        # Content-Security-Policy: only load scripts/styles from this server, block framing.
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def allowed(self):
        """Accept only requests addressed to localhost (blocks DNS-rebinding attacks)."""
        return self.headers.get('Host') in (f'127.0.0.1:{PORT}', f'localhost:{PORT}')

    def do_GET(self):
        """Handle page loads, the document list, and the model status check."""
        if not self.allowed():
            return self.send(403, {'error': 'Invalid host.'})
        try:
            if self.path == '/api/documents':
                return self.send(200, documents())
            if self.path == '/api/status':
                return self.send(200, status())
            if self.path not in STATIC:
                return self.send(404, {'error': 'Not found.'})
            filename, mime = STATIC[self.path]
            return self.send(200, (ROOT / 'static' / filename).read_bytes(), mime)
        except Exception as exc:
            self.send(400, {'error': str(exc)})

    def do_POST(self):
        """Handle upload, ask and delete requests."""
        origin = self.headers.get('Origin')
        # CSRF protection: require the right host, a custom header a web page on another
        # site cannot add without permission, and (if present) a same-origin Origin.
        if not self.allowed() or self.headers.get('X-Local-App') != 'document-qa' \
                or origin not in (None, f'http://localhost:{PORT}', f'http://127.0.0.1:{PORT}'):
            return self.send(403, {'error': 'Local requests only.'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 15 * 1024 * 1024:  # base64 inflates files by about a third
                raise ValueError('Request too large or empty.')
            data = json.loads(self.rfile.read(length))
            # Route by path to the matching action.
            if self.path == '/api/upload':
                result = upload(data['name'], base64.b64decode(data['content'], validate=True))
            elif self.path == '/api/ask':
                result = ask(data['question'])
            elif self.path == '/api/delete':
                delete(int(data['id']))
                result = {'deleted': True}
            else:
                return self.send(404, {'error': 'Not found.'})
            self.send(200, result)
        except Exception as exc:
            self.send(400, {'error': str(exc)})


# --- Entry point -------------------------------------------------------------
if __name__ == '__main__':
    print(f'Document Q&A: http://127.0.0.1:{PORT} — model {ANSWER_MODEL}')
    print('Keep this terminal open. Press Ctrl+C to stop.')
    try:
        # Bind to 127.0.0.1 only, so other computers on the network cannot connect.
        ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')