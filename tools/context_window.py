"""Read the context window the way the scheduler must: index, then required files only."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "docs" / "context"
INDEX = WINDOW / "INDEX.md"
REQUIRED = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def section(text, title):
    marker = "## " + title
    start = text.find(marker)
    if start < 0:
        raise SystemExit("missing section " + title)
    rest = text[start + len(marker):]
    nxt = rest.find("\n## ")
    return rest if nxt < 0 else rest[:nxt]


def required_paths(index_text):
    body = section(index_text, "必读")
    paths = []
    for link in REQUIRED.findall(body):
        path = (WINDOW / link).resolve()
        if WINDOW.resolve() not in path.parents:
            raise SystemExit("required path leaves the window: " + link)
        if "archive" in path.parts:
            raise SystemExit("required list points into archive: " + link)
        paths.append(path)
    if not paths:
        raise SystemExit("required list is empty")
    return paths


def load_window():
    index_text = INDEX.read_text(encoding="utf-8")
    opened = [INDEX]
    files = {}
    for path in required_paths(index_text):
        opened.append(path)
        files[path.name] = path.read_text(encoding="utf-8")
    if any("archive" in path.parts for path in opened):
        raise SystemExit("archive was opened")
    return opened, files


def main():
    opened, files = load_window()
    dispatch = json.loads(files["dispatch.json"])
    expected = {
        "schema", "stop", "ask_gpt", "gpt_conversation", "codex_session",
        "codex_model", "prompt_file", "next_command", "result_file",
    }
    if set(dispatch) != expected or dispatch["schema"] != "ref2dex.context-dispatch.v1":
        raise SystemExit("dispatch schema mismatch")
    if dispatch["codex_model"] != "gpt-6.1-sol":
        raise SystemExit("codex model is fixed")
    if dispatch["ask_gpt"] and not dispatch["gpt_conversation"]:
        raise SystemExit("ask_gpt requires the standing conversation")
    if dispatch["codex_session"] is None and dispatch["prompt_file"]:
        raise SystemExit("codex prompt without a session")
    if dispatch["stop"]:
        action = "stop"
    elif dispatch["ask_gpt"]:
        action = "ask_gpt"
    elif dispatch["next_command"]:
        action = "command"
    elif dispatch["codex_session"] and dispatch["prompt_file"]:
        action = "codex"
    else:
        raise SystemExit("dispatch has no action")
    print(json.dumps({
        "opened": [str(path.relative_to(ROOT)).replace("\\", "/") for path in opened],
        "archive_opened": False,
        "action": action,
        "stop": dispatch["stop"],
        "ask_gpt": dispatch["ask_gpt"],
        "codex_session": dispatch["codex_session"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
