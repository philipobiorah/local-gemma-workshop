// --- Small helpers ----------------------------------------------------------

// Shortcut: find an element by its id.
const $ = id => document.getElementById(id);

// Build an element with TEXT content (textContent, never innerHTML, so document
// text can't inject HTML or scripts). Optional CSS class as the third argument.
function node(tag, text, cls) {
  const n = document.createElement(tag);
  n.textContent = text;
  if (cls) n.className = cls;
  return n;
}

// Call the local server. With no data it is a GET; with data it is a JSON POST
// carrying the X-Local-App header the server requires. Throws on HTTP errors.
async function api(path, data) {
  const r = await fetch('/api/' + path, data === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Local-App': 'document-qa' },
    body: JSON.stringify(data)
  });
  const result = await r.json();
  if (!r.ok) throw Error(result.error);
  return result;
}

// --- Document list ----------------------------------------------------------

// Reload the list of indexed documents, each with a Remove button.
async function refresh() {
  const docs = await api('documents');
  $('documents').replaceChildren();  // clear the old list
  for (const doc of docs) {
    const li = node('li', `${doc.name} · ${doc.chunks} chunks`);
    const b = node('button', 'Remove');
    b.onclick = async () => {
      try { await api('delete', { id: doc.id }); await refresh(); }
      catch (e) { $('error').textContent = e.message; }
    };
    li.append(b);
    $('documents').append(li);
  }
}

// --- Upload -----------------------------------------------------------------

// Read a file and return its contents as base64 text (the part after "data:...;base64,").
function base64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(file);
  });
}

// "Upload and index" button: send each chosen file to the server one at a time.
$('upload').onclick = async () => {
  const files = [...$('files').files];
  if (!files.length) { $('progress').textContent = 'Choose a document first.'; return; }
  $('upload').disabled = true;  // prevent double clicks while indexing
  try {
    for (const file of files) {
      if (file.size > 10 * 1024 * 1024) throw Error('Document exceeds 10 MiB.');
      $('progress').textContent = `Indexing ${file.name}… Please wait.`;
      await api('upload', { name: file.name, content: await base64(file) });
      await refresh();  // show the new document straight away
    }
    $('progress').textContent = 'Documents indexed. Ask a question.';
    $('files').value = '';  // clear the file picker
  } catch (e) {
    $('progress').textContent = e.message;
  } finally {
    $('upload').disabled = false;
  }
};

// --- Asking a question ------------------------------------------------------

// "Ask Gemma" form: send the question and render the answer as a card.
$('ask').onsubmit = async event => {
  event.preventDefault();  // stop the browser reloading the page
  $('submit').disabled = true;
  $('error').textContent = '';
  const question = $('question').value;
  try {
    const r = await api('ask', { question });
    // Card header: the question and the model's answer.
    const card = node('article', '', 'answer');
    card.append(node('h3', question), node('p', r.answer));
    if (r.warning) card.append(node('p', r.warning, 'warning'));

    // Citations that passed server-side verification, with their source file/page.
    card.append(node('h4', 'Verified quotations'));
    if (!r.citations.length) card.append(node('p', 'No verified quotations.'));
    for (const c of r.citations) {
      const src = r.evidence.find(e => e.id === c.id);
      card.append(
        node('p', `[${c.id}] ${src.filename}${src.page ? ' · page/section ' + src.page : ''}`),
        node('blockquote', c.quote));
    }

    // Collapsible list of every passage the model saw, so you can check its work.
    const detail = node('details', '');
    detail.append(node('summary', 'Inspect all retrieved passages'));
    for (const e of r.evidence) {
      detail.append(
        node('h4', `[${e.id}] ${e.filename}${e.page ? ' · page/section ' + e.page : ''}`),
        node('blockquote', e.text));
    }
    card.append(detail, node('p', 'Model: ' + r.model, 'muted'));
    $('results').prepend(card);  // newest answer first
  } catch (e) {
    $('error').textContent = e.message;
  } finally {
    $('submit').disabled = false;
  }
};

// --- Startup ----------------------------------------------------------------

// On page load: show existing documents and report whether both models are installed.
async function start() {
  try {
    await refresh();
    const s = await api('status');
    $('status').textContent = s.ready
      ? `Ready · ${s.model} + ${s.embedding_model}`
      : 'Models missing. Run: ollama pull gemma4:e2b && ollama pull embeddinggemma:300m, then refresh.';
  } catch (e) {
    $('status').textContent = e.message;
  }
}
start();