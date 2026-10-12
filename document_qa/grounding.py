"""Reading and checking the model's answer. Pure functions: no ADK, no database."""
import json
import re

NOT_FOUND = 'I could not find that in the uploaded documents.'
UNVERIFIED = ('I could not produce an answer with verified source quotations. '
              'Review the retrieved passages or rephrase your question.')


def format_passages(passages):
    """Render passages as plain text with clear delimiters, for the model's prompt."""
    return '\n\n'.join(
        f"--- passage id={p['id']} file={p['filename']} page={p['page'] or 'n/a'} ---\n{p['text']}"
        for p in passages)


def parse_answer(text):
    """Extract the JSON object from the model's reply and return it as a dict."""
    text = text.strip()
    # Small models add prose or code fences around the JSON; take the outermost object.
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end <= start:
        raise ValueError('Gemma did not return valid answer JSON. Rephrase the question and retry.')
    try:
        result = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError('Gemma did not return valid answer JSON. Rephrase the question and retry.') from exc
    if not isinstance(result, dict):
        raise ValueError('The answer must be a JSON object.')
    return result


def verify_answer(result, passages):
    """Keep only citations whose quote really appears in a retrieved passage.

    Returns {'answer', 'citations', 'warning'}; unsupported answers are replaced
    by a notice, and a genuine "not found" answer is allowed through.
    """
    answer, citations = result.get('answer'), result.get('citations')
    if not isinstance(answer, str) or not isinstance(citations, list):
        raise ValueError('Gemma returned an invalid answer format.')
    lookup = {p['id']: p for p in passages}  # passage id -> passage
    valid = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        ident, quote = citation.get('id'), citation.get('quote')
        if isinstance(ident, str) and ident.strip().isdigit():
            ident = int(ident)  # small models often return ids as strings
        if isinstance(quote, str):
            quote = re.sub(r'\s+', ' ', quote).strip()  # passages are whitespace-normalised
        # Valid only if: the id was retrieved, the quote is 8+ chars, and it is an exact substring.
        if isinstance(ident, int) and ident in lookup and isinstance(quote, str) \
                and len(quote) >= 8 and quote in lookup[ident]['text']:
            valid.append({'id': ident, 'quote': quote})
    warning = None
    if citations and len(valid) != len(citations):
        warning = 'Some generated citations failed verification. Check the passages before using this answer.'
    not_found = not citations and answer.strip() == NOT_FOUND
    if not valid and not not_found:
        answer = UNVERIFIED  # never show an answer without verified evidence
    return {'answer': answer, 'citations': valid, 'warning': warning}


def to_markdown(checked, passages, notes=()):
    """Format a verified answer for the adk web chat window."""
    lines = [checked['answer']]
    if checked['warning']:
        lines.append(f"\n*{checked['warning']}*")
    if checked['citations']:
        lines.append('\n**Verified quotations**')
        by_id = {p['id']: p for p in passages}
        for c in checked['citations']:
            source = by_id[c['id']]
            where = f" · page/section {source['page']}" if source['page'] else ''
            lines.append(f"- [{c['id']}] {source['filename']}{where}: \"{c['quote']}\"")
    if notes:
        lines.append('\n*Notes: ' + '; '.join(notes) + '*')
    return '\n'.join(lines)