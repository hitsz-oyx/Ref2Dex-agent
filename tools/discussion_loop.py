"""One discussion cycle. It waits for a named result, sends that result, then resumes Codex."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
LOOP = ROOT / "docs" / "context" / "loop.json"
STATE = ROOT / "docs" / "context" / "loop-state.json"
GPT = TOOLS / "gpt_continue.mjs"
HERMES = Path(r"D:\Hermes\hermes-agent\.hermes\bin\hermes.exe")
CHAT = "feishu:oc_90c411e7a20ac2964232bd6d6e130dff"
ALLOW = ("REPRODUCED", "NOT_REPRODUCED")


def load_loop():
    loop = json.loads(LOOP.read_text(encoding="utf-8"))
    expected = {
        "schema", "enabled", "gpt_conversation", "codex_session", "codex_home",
        "codex_model", "codex_done_line", "wait", "thresholds_mm", "prompts",
    }
    if set(loop) != expected or loop["schema"] != "ref2dex.discussion-loop.v1":
        raise SystemExit("loop schema mismatch")
    if loop["codex_model"] != "gpt-6.1-sol":
        raise SystemExit("codex model is fixed")
    if set(loop["prompts"]) != set(ALLOW):
        raise SystemExit("loop prompts do not match the allow list")
    if Path(loop["wait"]["directory"]).name != "C2-matched-control":
        raise SystemExit("loop is pinned to the control directory")
    return loop


def reproduced(delta, thresholds):
    if len(delta) != 3 or len(thresholds) != 3:
        raise SystemExit("delta and thresholds must have three values")
    return all(float(value) >= float(limit) for value, limit in zip(delta, thresholds))


def first_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip().strip("`*")
    return ""


def fmt(values):
    return " / ".join(f"{float(value):+.3f}" for value in values)


def fetch(loop):
    directory = loop["wait"]["directory"]
    code = r"""
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
summary = json.loads((directory / "panel-step000500" / "summary.json").read_text())
result = json.loads((directory / "result.json").read_text())
recipe = json.loads((directory / "recipe-v2.json").read_text())
versus = summary["versus_local3000"]
flow = result.get("flow_preclip_grad_norm") or []
trunk = result.get("trunk_preclip_grad_norm") or []
print(json.dumps({
    "status": result["status"],
    "updates": result["updates"],
    "means_mm": summary["means_mm"],
    "delta_mm": versus["delta_mm"],
    "ci95_mm": versus["paired_mother_cluster_ci95_mm"],
    "matched_clip": recipe.get("matched_clip"),
    "flow_only": recipe.get("flow_only"),
    "flow_gt1": sum(value > 1 for value in flow),
    "trunk_gt1": sum(value > 1 for value in trunk),
    "flow_n": len(flow),
    "trunk_n": len(trunk),
}, ensure_ascii=False))
"""
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", loop["wait"]["host"], "python3", "-", directory],
        input=code, text=True, capture_output=True,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "result is not ready")
    measured = json.loads(proc.stdout)
    if measured["status"] != "COMPLETED" or measured["updates"] != 500:
        raise SystemExit("control run is not a completed 500-update result")
    if measured["matched_clip"] is not True or measured["flow_only"] is not False:
        raise SystemExit("control recipe switches are not the authorized pair")
    if measured["flow_n"] != 500 or measured["trunk_n"] != 500:
        raise SystemExit("control clip logs are incomplete")
    return measured


def packet(loop, measured, token, sha):
    ci = ", ".join(fmt(pair) for pair in measured["ci95_mm"])
    return "\n".join([
        "仓库 https://github.com/hitsz-oyx/Ref2Dex-agent",
        "提交 " + sha,
        "同一条对话。这是脚本从结果文件提取的对照臂，不是新实验。",
        "",
        "diagnosis/C2-matched-control 已完成 500 updates。matched_clip 开，flow_only 关。原点 [-2,-2,-2]，体素 0.005。",
        "means " + fmt(measured["means_mm"]) + " mm。",
        "相对 local3000 " + fmt(measured["delta_mm"]) + " mm。",
        "母序列 CI95 " + ci + "。",
        "复现门槛是三项都 ≥ +1.827 / +7.059 / +0.855 mm。脚本判定 " + token + "。",
        "flow_future 与 trunk 的 pre-clip 范数大于 1：" + f"{measured['flow_gt1']}/500 和 {measured['trunk_gt1']}/500。这不是结果门。",
        "",
        "第一行只能是 REPRODUCED 或 NOT_REPRODUCED。",
        "不要提出开跑 flow-only。这一轮没有授权。",
        "",
    ])


def measured_block(measured, token):
    ci = ", ".join(fmt(pair) for pair in measured["ci95_mm"])
    return "\n".join([
        "MEASURED",
        "directory: diagnosis/C2-matched-control",
        "updates: 500",
        "means_mm: " + fmt(measured["means_mm"]),
        "delta_vs_local3000_mm: " + fmt(measured["delta_mm"]),
        "ci95_mm: " + ci,
        "thresholds_mm: +1.827 / +7.059 / +0.855",
        "script_token: " + token,
        "flow_preclip_gt1: %d/500" % measured["flow_gt1"],
        "trunk_preclip_gt1: %d/500" % measured["trunk_gt1"],
        "",
    ])


def notify(text):
    path = ROOT / "tmp" / "loop-feishu.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    subprocess.run([str(HERMES), "send", "--to", CHAT, "--file", str(path), "--json"], check=False)


def send_gpt(loop, body):
    out = ROOT / "tmp" / "loop-packet.txt"
    reply = ROOT / "tmp" / "loop-gpt-reply.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(body, encoding="utf-8")
    proc = subprocess.run(
        ["node", str(GPT), loop["gpt_conversation"], str(out), "--wait-reply", str(reply)],
        cwd=ROOT, text=True,
    )
    if proc.returncode != 0 or not reply.exists():
        raise SystemExit("gpt continue failed")
    return reply.read_text(encoding="utf-8")


def codex_command(args):
    found = shutil.which("codex")
    if found and found.lower().endswith(".cmd"):
        script = Path(found).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        if script.exists():
            return ["node", str(script), *args]
    if not found:
        raise SystemExit("codex is not on PATH")
    return [found, *args]


def resume_codex(loop, prompt):
    last = ROOT / "tmp" / "loop-codex-last.txt"
    env = os.environ.copy()
    env.pop("CODEX_API_KEY", None)
    env["CODEX_HOME"] = loop["codex_home"]
    log = ROOT / "tmp" / "loop-codex.jsonl"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            codex_command(["exec", "resume", "--json", "-m", loop["codex_model"],
                           "--dangerously-bypass-approvals-and-sandbox", "-o", str(last),
                           loop["codex_session"], "-"]),
            input=prompt, text=True, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise SystemExit("codex resume failed")
    got = last.read_text(encoding="utf-8").strip().splitlines()
    if not got or got[-1].strip() != loop["codex_done_line"]:
        raise SystemExit("codex did not finish with the done line")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    loop = load_loop()
    if not loop["enabled"]:
        raise SystemExit("loop is disabled")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    if state.get("phase") == "codex_done" and state.get("directory") == loop["wait"]["directory"]:
        print(json.dumps({"action": "already_done", "token": state.get("token")}, ensure_ascii=False))
        return
    measured = fetch(loop)
    token = "REPRODUCED" if reproduced(measured["delta_mm"], loop["thresholds_mm"]) else "NOT_REPRODUCED"
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    body = packet(loop, measured, token, sha)
    report = {
        "token": token,
        "delta_mm": [round(float(value), 3) for value in measured["delta_mm"]],
        "packet_chars": len(body),
        "sent": False,
    }
    if not args.execute:
        print(json.dumps(report, ensure_ascii=False))
        return
    if state.get("phase") == "gpt_accepted" and state.get("directory") == loop["wait"]["directory"]:
        reply_token = state["token"]
    else:
        reply = send_gpt(loop, body)
        reply_token = first_line(reply)
        report["sent"] = True
        report["reply_token"] = reply_token
        if reply_token != token:
            notify("讨论循环停住。脚本判定 %s，GPT 第一行是 %s。\n%s" % (token, reply_token or "空", reply[:500]))
            print(json.dumps(report, ensure_ascii=False))
            raise SystemExit("gpt reply is not the script token")
        STATE.write_text(json.dumps({
            "phase": "gpt_accepted", "directory": loop["wait"]["directory"], "token": token,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prompt = (ROOT / loop["prompts"][reply_token]).read_text(encoding="utf-8")
    prompt += "\n" + measured_block(measured, reply_token)
    resume_codex(loop, prompt)
    STATE.write_text(json.dumps({
        "phase": "codex_done", "directory": loop["wait"]["directory"], "token": token,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["codex"] = "CARD_UPDATED"
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
