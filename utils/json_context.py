import json

_OMITTED_KEY = "__omitted__"


def _bound(value, max_items: int, max_str_len: int):
    if isinstance(value, dict):
        items = list(value.items())
        bounded = {k: _bound(v, max_items, max_str_len) for k, v in items[:max_items]}
        if len(items) > max_items:
            bounded[_OMITTED_KEY] = f"{len(items) - max_items} more keys omitted"
        return bounded

    if isinstance(value, list):
        bounded = [_bound(v, max_items, max_str_len) for v in value[:max_items]]
        if len(value) > max_items:
            bounded.append(f"... {len(value) - max_items} more items omitted")
        return bounded

    if isinstance(value, str) and len(value) > max_str_len:
        return value[:max_str_len] + f"... [truncated, {len(value)} chars total]"

    return value


def summarize_json(data, max_items: int = 15, max_str_len: int = 300, max_chars: int = 6000) -> str:
    """Render `data` as JSON bounded to a size safe for embedding in an LLM prompt.

    Recursively caps list/dict length and string length so the result stays
    valid, readable JSON at every truncation point, with a final hard
    character ceiling as a backstop for pathological single values.
    """
    bounded = _bound(data, max_items, max_str_len)
    text = json.dumps(bounded, indent=2, default=str)

    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated, {len(text)} chars total]"

    return text
