"""One scheduler step. It copies the dispatch card and does not invent a message."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from context_window import ROOT, load_window

REPO = "https://github.com/hitsz-oyx/Ref2Dex-agent"
GPT_CONTINUE = TOOLS / "gpt_continue.mjs"


def decide(dispatch):
    if dispatch["codex_model"] != "gpt-6.1-sol":
        raise SystemExit("codex model is fixed")
    if dispatch["ask_gpt"] and not str(dispatch.get("gpt_conversation") or "").startswith("https://chatgpt.com/c/"):
        raise SystemExit("ask_gpt requires the standing conversation")
    if dispatch["codex_session"] is None and dispatch["prompt_file"]:
        raise SystemExit("codex prompt without a session")
    if dispatch["stop"]:
        return "stop"
    if dispatch["ask_gpt"]:
        return "ask_gpt"
    if dispatch["next_command"]:
        return "command"
    if dispatch["codex_session"] and dispatch["prompt_file"]:
        return "codex"
    raise SystemExit("dispatch has no action")


def packet(dispatch, recent):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return "\n".join([
        "[调度单]",
        "仓库: " + REPO,
        "提交: " + sha,
        "先读: docs/context/INDEX.md 里标成必读的文件。不要打开归档。",
        "GPT 对话: " + str(dispatch["gpt_conversation"]),
        "recent:",
        recent.strip(),
        "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--check-gpt", action="store_true")
    args = parser.parse_args()
    opened, files = load_window()
    dispatch = json.loads(files["dispatch.json"])
    action = decide(dispatch)
    body = packet(dispatch, files["recent.md"])
    if "archive/" in body:
        raise SystemExit("packet includes the archive")
    sent = False
    gpt_check = None
    if args.check_gpt:
        url = str(dispatch["gpt_conversation"])
        proc = subprocess.run(["node", str(GPT_CONTINUE), url, "--check"], cwd=ROOT, text=True, capture_output=True)
        gpt_check = "ok" if proc.returncode == 0 and "COMPOSER_OK" in proc.stdout else "fail"
        if gpt_check == "fail":
            sys.stderr.write((proc.stdout or "") + (proc.stderr or ""))
    if args.execute and action == "ask_gpt":
        out = ROOT / "tmp" / "gpt-packet.txt"
        out.parent.mkdir(exist_ok=True)
        out.write_text(body, encoding="utf-8")
        proc = subprocess.run(["node", str(GPT_CONTINUE), dispatch["gpt_conversation"], str(out)], cwd=ROOT)
        if proc.returncode != 0:
            raise SystemExit("gpt continue failed")
        sent = True
    if args.execute and action == "codex" and not dispatch.get("codex_home"):
        raise SystemExit("codex home is unset")
    print(json.dumps({
        "opened": [str(path.relative_to(ROOT)).replace("\\", "/") for path in opened],
        "archive_opened": False,
        "action": action,
        "sent": sent,
        "gpt_check": gpt_check,
        "codex_session": dispatch["codex_session"],
        "packet_chars": len(body),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
