"""Local document index: files folder -> text -> chunks -> embeddings -> SQLite.

The `documents/` folder is the single source of truth. Every question first
calls Store.sync(), which indexes new files and forgets deleted ones, so
adding or removing a file is all it takes to change what the agent can see.
"""
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import urllib.request
from contextlib import contextmanager

# --- Configuration -----------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = PACKAGE_DIR / 'documents'             # drop files here
DB_PATH = PACKAGE_DIR / 'data' / 'index.sqlite3'      # created automatically
EMBED_MODEL = 'embeddinggemma:300m'                   # model that turns text into vectors
OLLAMA_URL = 'http://127.0.0.1:11434'
SUPPORTED = ('.pdf', '.docx', '.txt', '.md')
MAX_FILE = 10 * 1024 * 1024                           # 10 MiB per document
MAX_CHUNKS = 2000                                     # total across all documents
# When a chat attachment has no usable file name, its MIME type picks the extension.
MIME_SUFFIX = {
    'application/pdf': '.pdf',
    'text/plain': '.txt',
    'text/markdown': '.md',
    'text/x-markdown': '.md',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
}
HELP = f'Add PDF, DOCX, TXT or Markdown files to {DOCUMENTS_DIR} or attach them in the chat.'


# --- Ollama embeddings -------------------------------------------------------
class Ollama:
    """Tiny HTTP client for the local Ollama server (embeddings only)."""

    def __init__(self):
        # An empty ProxyHandler stops system proxy settings from intercepting localhost calls.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def installed(self):
        """Names of the models installed in Ollama (used by app.py's status line)."""
        try:
            with self.opener.open(OLLAMA_URL + '/api/tags', timeout=10) as response:
                return [m['name'] for m in json.load(response).get('models', [])]
        except Exception as exc:
            raise ValueError('Ollama request failed. Start Ollama and check the models. ' + str(exc)) from exc

    def embed(self, texts):
        """Turn a list of texts into a list of vectors (lists of numbers)."""
        # truncate=False errors instead of silently cutting text.
        payload = {'model': EMBED_MODEL, 'input': texts, 'truncate': False, 'keep_alive': '5m'}
        request = urllib.request.Request(
            OLLAMA_URL + '/api/embed', data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=600) as response:
                vectors = json.load(response).get('embeddings', [])
        except Exception as exc:
            raise ValueError('Ollama request failed. Start Ollama and check the models. ' + str(exc)) from exc
        # Sanity check: exactly one non-empty vector per input text.
        if len(vectors) != len(texts) or any(not v for v in vectors):
            raise ValueError('Ollama returned invalid embeddings.')
        return vectors


# --- Text extraction and chunking -------------------------------------------
def extract(name, content):
    """Read a file's bytes and return a list of (page_number, text) pairs."""
    suffix = Path(name).suffix.lower()
    if suffix in ('.txt', '.md'):
        pages = [(1, content.decode('utf-8-sig'))]   # no pages in plain text; utf-8-sig strips a BOM
    elif suffix == '.pdf':
        from pypdf import PdfReader  # imported here so it is only needed for PDFs
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            raise ValueError('Encrypted PDFs are not supported. Use an unlocked copy.')
        if len(reader.pages) > 200:
            raise ValueError('Limit: 200 PDF pages per document.')
        # One (page, text) pair per page; image-only pages give empty text.
        pages = [(i + 1, page.extract_text() or '') for i, page in enumerate(reader.pages)]
    elif suffix == '.docx':
        from docx import Document
        doc = Document(io.BytesIO(content))
        # Paragraphs first, then each table row with cells joined by " | ".
        text = '\n'.join([p.text for p in doc.paragraphs] +
                         [' | '.join(c.text for c in row.cells) for t in doc.tables for row in t.rows])
        pages = [(None, text)]  # Word files have no fixed page numbers
    else:
        raise ValueError('Use PDF, DOCX, UTF-8 TXT or Markdown.')
    if not any(t.strip() for _, t in pages):
        raise ValueError('No readable text found. Scanned PDFs need OCR first.')
    if sum(len(t) for _, t in pages) > 500000:
        raise ValueError('Limit: 500,000 extracted characters per document.')
    return pages


def chunks(pages):
    """Split each page into ~1000 character pieces that overlap by 150 characters."""
    out = []
    for page, text in pages:
        text = re.sub(r'\s+', ' ', text).strip()  # collapse all whitespace into single spaces
        start = 0
        while start < len(text):
            end = min(start + 1000, len(text))
            if end < len(text):
                # Prefer to cut at a space between 650 and 1000 chars so words are not split.
                boundary = text.rfind(' ', start + 650, end)
                if boundary > start:
                    end = boundary
            out.append((page, text[start:end]))
            if end == len(text):
                break
            start = end - 150  # overlap, so a sentence cut at the boundary appears in both chunks
    return out


def cosine(a, b):
    """Cosine similarity: 1 means same direction (very similar), 0 means unrelated."""
    if len(a) != len(b):
        raise ValueError('Embedding dimensions changed. Delete data/index.sqlite3 to re-index.')
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denominator if denominator else 0


# --- The index ---------------------------------------------------------------
class Store:
    """SQLite-backed index of the files in the documents folder."""

    def __init__(self, db_path=DB_PATH, documents_dir=DOCUMENTS_DIR, embedder=None):
        self.db_path = Path(db_path)
        self.documents_dir = Path(documents_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Ollama()  # swap in a fake for tests
        self._lock = threading.Lock()         # one sync at a time
        self._digests = {}                    # file name -> ((mtime, size), sha256), avoids re-hashing
        with self._db() as db:
            # documents: one row per file (digest = SHA-256 of its bytes).
            # chunks: text pieces plus their vectors stored as JSON text.
            # ON DELETE CASCADE removes a document's chunks with the document.
            db.executescript('''CREATE TABLE IF NOT EXISTS documents
                (id INTEGER PRIMARY KEY, name TEXT, digest TEXT UNIQUE);
                CREATE TABLE IF NOT EXISTS chunks
                (id INTEGER PRIMARY KEY, document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                 page INTEGER, text TEXT, vector TEXT);''')

    @contextmanager
    def _db(self):
        """Open a connection, commit on success (roll back on error), always close it."""
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row            # access columns by name
        db.execute('PRAGMA foreign_keys=ON')    # SQLite ignores foreign keys unless enabled
        try:
            with db:
                yield db
        finally:
            db.close()

    def documents(self):
        """List every indexed document with its id, name and number of chunks."""
        with self._db() as db:
            return [dict(r) for r in db.execute(
                'SELECT d.id,d.name,count(c.id) AS chunks FROM documents d '
                'LEFT JOIN chunks c ON d.id=c.document_id GROUP BY d.id')]

    # -- folder handling -------------------------------------------------------
    def add_file(self, name, data, mime=None):
        """Save a chat attachment into the documents folder and return its file name.

        The next sync() indexes it. Raises ValueError for unsupported or oversized files.
        """
        if len(data) > MAX_FILE:
            raise ValueError('larger than 10 MiB')
        # Keep only a plain file name: no folders, no odd characters, no leading dots.
        name = re.sub(r'[^\w.\- ]', '_', re.split(r'[/\\]', name or '')[-1]).strip('. ')[:120] or 'attachment'
        if Path(name).suffix.lower() not in SUPPORTED:
            suffix = MIME_SUFFIX.get((mime or '').split(';')[0].strip().lower())
            if not suffix:
                raise ValueError(f'unsupported file type {mime or Path(name).suffix or "unknown"}')
            name = Path(name).stem + suffix
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        target = self.documents_dir / name
        if target.exists() and target.read_bytes() != data:
            # Same name, different content: keep both by adding a short content hash.
            target = target.with_name(f'{target.stem}-{hashlib.sha256(data).hexdigest()[:8]}{target.suffix}')
        if not target.exists():
            target.write_bytes(data)
        return target.name

    def remove_file(self, name):
        """Delete a file from the documents folder; the next sync() forgets it."""
        if name != Path(name).name:
            raise ValueError('Invalid file name.')  # plain names only: never leave the folder
        (self.documents_dir / name).unlink(missing_ok=True)

    def _digest(self, path):
        """SHA-256 of a file, re-read only when its size or modified time changed."""
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._digests.get(path.name)
        if not cached or cached[0] != key:
            cached = (key, hashlib.sha256(path.read_bytes()).hexdigest())
            self._digests[path.name] = cached
        return cached[1]

    def sync(self):
        """Make the index match the documents folder. Returns a list of problems (strings)."""
        with self._lock:
            problems, found = [], {}   # found: content digest -> every file with that content
            if self.documents_dir.is_dir():
                for path in sorted(self.documents_dir.iterdir()):
                    if not path.is_file() or path.name.startswith('.') or path.suffix.lower() not in SUPPORTED:
                        continue
                    try:
                        if path.stat().st_size > MAX_FILE:
                            raise ValueError('larger than 10 MiB')
                        found.setdefault(self._digest(path), []).append(path)
                    except (OSError, ValueError) as exc:
                        problems.append(f'{path.name}: {exc}')
            with self._db() as db:
                indexed = {r['digest']: r for r in db.execute('SELECT id,name,digest FROM documents')}
                # Identical files count once: keep the name already indexed if that file
                # still exists, otherwise the first name alphabetically.
                files = {d: next((p for p in paths if d in indexed and p.name == indexed[d]['name']), paths[0])
                         for d, paths in found.items()}
                for digest, row in indexed.items():
                    if digest not in files:   # file was deleted: forget it
                        db.execute('DELETE FROM documents WHERE id=?', (row['id'],))
                    elif row['name'] != files[digest].name:  # file was renamed
                        db.execute('UPDATE documents SET name=? WHERE id=?', (files[digest].name, row['id']))
            for digest, path in files.items():
                if digest not in indexed:
                    try:
                        self._index(path, digest)
                    except Exception as exc:  # one bad file must not block the others
                        problems.append(f'{path.name}: {exc}')
            return problems

    def _index(self, path, digest):
        """Extract, chunk, embed and save one file."""
        parts = chunks(extract(path.name, path.read_bytes()))
        with self._db() as db:
            total = db.execute('SELECT count(*) FROM chunks').fetchone()[0]
        if total + len(parts) > MAX_CHUNKS:
            raise ValueError(f'limit of {MAX_CHUNKS} chunks reached; remove a document first')
        vectors = []
        # Embed 8 chunks per request. The file name is prefixed to give the embedding context.
        for offset in range(0, len(parts), 8):
            vectors.extend(self.embedder.embed(
                ['title: ' + path.name + ' | text: ' + t for _, t in parts[offset:offset + 8]]))
        # Save the document and all its chunks in one transaction.
        with self._db() as db:
            ident = db.execute('INSERT INTO documents(name,digest) VALUES (?,?)', (path.name, digest)).lastrowid
            db.executemany('INSERT INTO chunks(document_id,page,text,vector) VALUES (?,?,?,?)',
                           [(ident, p, t, json.dumps(v)) for (p, t), v in zip(parts, vectors)])

    # -- search ----------------------------------------------------------------
    def search(self, query):
        """Return the best passages for a query: top 5 by similarity plus each document's opening."""
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError('Enter a question of 1 to 1,000 characters.')
        with self._db() as db:
            rows = list(db.execute('SELECT c.*,d.name FROM chunks c JOIN documents d ON d.id=c.document_id'))
        if not rows:
            raise ValueError('No documents are indexed yet. ' + HELP)
        # Embed the query, then rank every chunk by cosine similarity and keep the top 5.
        vector = self.embedder.embed(['task: search result | query: ' + query])[0]
        ranked = sorted(rows, key=lambda r: cosine(vector, json.loads(r['vector'])), reverse=True)[:5]
        # Title/author questions target the opening of a document, which rarely
        # ranks by similarity. Always include each document's first chunk (up to 3 documents).
        firsts = {}
        for r in rows:
            if r['document_id'] not in firsts or r['id'] < firsts[r['document_id']]['id']:
                firsts[r['document_id']] = r
        chosen = list(ranked)
        for r in list(firsts.values())[:3]:
            if r['id'] not in {c['id'] for c in chosen}:
                chosen.append(r)
        return [{'id': r['id'], 'filename': r['name'], 'page': r['page'], 'text': r['text']} for r in chosen]


# --- Shared instance ---------------------------------------------------------
_store = None


def get_store():
    """The one Store the agent uses, created on first use (importing has no side effects)."""
    global _store
    if _store is None:
        _store = Store()
    return _store


def set_store(store):
    """Replace the shared Store (used by tests)."""
    global _store
    _store = store