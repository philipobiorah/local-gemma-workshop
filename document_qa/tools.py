"""Tools the agent may call. The first search is forced in callbacks.py; this is the optional second."""
import asyncio

from google.adk.tools import ToolContext

from .store import get_store


async def search_documents(query: str, tool_context: ToolContext) -> dict:
    """Search the uploaded documents again with a different query.

    Use this only if the passages already provided do not answer the question.

    Args:
        query: A short natural-language search question.

    Returns:
        passages: IDs, filenames, page references and exact source text.
    """
    # ADK reads this signature and docstring to describe the tool to the model;
    # tool_context is supplied by ADK and hidden from the model.
    state = tool_context.state
    state['temp:extra_searches'] = used = state.get('temp:extra_searches', 0) + 1
    if used > 1:  # allow only one extra search per question
        return {'error': 'Search limit reached. Answer using the existing passages.'}
    try:
        found = await asyncio.to_thread(get_store().search, query)  # keep the event loop free
    except ValueError as exc:
        return {'error': str(exc)}  # report errors to the model instead of crashing
    # Remember the new passages so their quotes can be verified later.
    known = state.get('temp:passages', [])
    seen = {p['id'] for p in known}
    state['temp:passages'] = known + [p for p in found if p['id'] not in seen]
    return {'passages': found}