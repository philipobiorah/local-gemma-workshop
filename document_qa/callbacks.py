"""ADK callbacks: they run around every model call.

before_model (runs first):
  1. Caps the number of model calls per question.
  2. Saves chat attachments into the documents folder and hides the raw bytes
     from the model (a local model cannot read a PDF sent as binary).
  3. Syncs the documents folder into the index.
  4. Searches in code and shows the passages to the model, because small local
     models often skip optional tool calls.

after_model (runs on the model's reply):
  Checks every quotation against the retrieved passages and rewrites the JSON
  into a readable answer with only the verified quotations.

Per-question data lives in `temp:` state, which ADK discards after each question.
"""
import asyncio

from google.adk.models.llm_response import LlmResponse
from google.genai import types

from .grounding import format_passages, parse_answer, to_markdown, verify_answer
from .store import HELP, get_store

MAX_MODEL_CALLS = 4  # one answer, plus one optional extra search and its answer


def _say(text, data=None):
    """A ready-made model reply: ADK skips the real model call when a callback returns one.

    `data` rides along as event metadata, so other front ends (app.py) can read the
    structured answer instead of parsing the markdown text.
    """
    return LlmResponse(content=types.Content(role='model', parts=[types.Part(text=text)]),
                       custom_metadata={'document_qa': data} if data else None)


def _hide_attachments(contents, store, notes):
    """Return (contents, question): attachments saved and replaced by text, plus the newest question.

    The session keeps the original attachment bytes, so this runs on every model
    call, working on copies so the stored history is never changed.
    """
    prepared, question = [], ''
    for content in contents:
        parts, text = [], ''
        for part in content.parts or []:
            blob = part.inline_data
            if blob is None:
                parts.append(part)
                text += part.text or ''
                continue
            try:
                name = store.add_file(blob.display_name, blob.data or b'', blob.mime_type)
                parts.append(types.Part(text=f'[Attached file: {name}]'))
            except ValueError as exc:
                label = blob.display_name or blob.mime_type or 'file'
                parts.append(types.Part(text=f'[Attachment skipped: {label}]'))
                note = f'attachment {label} skipped ({exc})'
                if note not in notes:
                    notes.append(note)
        prepared.append(types.Content(role=content.role, parts=parts) if content.parts else content)
        if content.role == 'user' and text.strip():
            question = text  # ends as the newest user message that has real text
    return prepared, question


async def before_model(callback_context, llm_request):
    state = callback_context.state

    # 1. Safety cap on model calls for this question.
    state['temp:llm_calls'] = calls = state.get('temp:llm_calls', 0) + 1
    if calls > MAX_MODEL_CALLS:
        return _say('I could not finish answering. Please rephrase the question and try again.')

    # 2. Attachments -> documents folder.
    store = get_store()
    notes = []
    contents, question = _hide_attachments(llm_request.contents, store, notes)
    llm_request.contents = contents

    # 3. Documents folder -> index (blocking work runs in a thread).
    problems = await asyncio.to_thread(store.sync)
    state['temp:notes'] = notes + [f'could not index {p}' for p in problems]

    # 4. Forced retrieval, once per question.
    if 'temp:context' not in state:
        if not question.strip():
            return _say('Attachment received and indexed. Ask a question about your documents.'
                        if any(c.role == 'user' for c in contents) and not notes else
                        'Ask a question about your documents. ' + HELP)
        try:
            passages = await asyncio.to_thread(store.search, question)
        except ValueError as exc:
            return _say(' '.join([str(exc)] + [f'({n})' for n in state['temp:notes']]))
        state['temp:passages'] = passages            # everything the model may cite
        state['temp:context'] = format_passages(passages)  # what the model is shown

    # The session history does not keep the passages, so add them to the newest
    # user message on every model call (copying it, not editing the stored one).
    for index in range(len(contents) - 1, -1, -1):
        original = contents[index]
        if original.role == 'user' and any(p.text for p in original.parts or []):
            contents[index] = types.Content(role='user', parts=list(original.parts) + [
                types.Part(text='\n\nRetrieved passages:\n' + state['temp:context'])])
            break
    return None


def after_model(callback_context, llm_response):
    parts = (llm_response.content.parts or []) if llm_response.content else []
    if llm_response.partial or not parts or any(p.function_call for p in parts):
        return None  # a streaming fragment or a tool call: let ADK carry on
    state = callback_context.state
    passages = state.get('temp:passages')
    if passages is None:
        return None  # nothing was retrieved for this question
    text = ''.join(p.text or '' for p in parts if not p.thought)
    try:
        checked = verify_answer(parse_answer(text), passages)
    except ValueError as exc:
        return _say(str(exc))
    notes = list(state.get('temp:notes', []))
    return _say(to_markdown(checked, passages, notes),
                {**checked, 'evidence': passages, 'notes': notes})