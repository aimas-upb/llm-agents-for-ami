"""
Generic workspace matching utilities.
"""



def workspace_match(pattern: str, workspace_id: str) -> bool:
    """
    Check if workspace matches pattern.

    Supports various matching patterns including URLs and workspace IDs.

    Args:
        pattern: The workspace pattern or ID to match
        workspace_id: The workspace ID to check against

    Returns:
        True if the workspace matches the pattern
    """
    if not workspace_id:
        return True
    if not pattern:
        return False

    ws = str(workspace_id).strip()
    s = str(pattern)

    if ws.startswith("http://") or ws.startswith("https://"):
        return s == ws or ws in s

    if s == ws:
        return True
    if f"/{ws}#" in s:
        return True
    if f"/{ws}/" in s:
        return True
    if s.endswith("/" + ws):
        return True
    return ws in s


