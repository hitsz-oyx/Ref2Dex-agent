"""One discussion cycle. It waits for a named result, sends that result, then resumes Codex."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
LOOP = ROOT / "docs" / "context" / "loop.json"
STATE = ROOT / "docs" / "context" / "loop-state.json"
GPT = TOOLS / "gpt_continue.mjs"
HERMES = Path(r"D:\Hermes\hermes-agent\.hermes\bin\hermes.exe")
CHAT = "feishu:oc_90c411e7a20ac2964232bd6d6e130dff"
ALLOW = ("PASS", "FAIL", "UNIDENTIFIED")


class NotReady(Exception):
    pass


def load_loop():
    loop = json.loads(LOOP.read_text(encoding="utf-8"))
    expected = {
        "schema", "enabled", "cycle", "gpt_conversation", "codex_session", "codex_home",
        "codex_model", "codex_done_line", "wait", "expect", "pass_max_mm",
        "pass_ci_hi_mm", "fail_min_mm", "prompts",
    }
    if set(loop) != expected or loop["schema"] != "ref2dex.discussion-loop.v1":
        raise SystemExit("loop schema mismatch")
    if loop["codex_model"] != "gpt-6.1-sol":
        raise SystemExit("codex model is fixed")
    if set(loop["prompts"]) != set(ALLOW):
        raise SystemExit("loop prompts do not match the allow list")
    if Path(loop["wait"]["directory"]).name != "C2-matched-flow":
        raise SystemExit("loop is pinned to the flow-only directory")
    if loop["expect"] != {"matched_clip": True, "flow_only": True}:
        raise SystemExit("flow-only switches are fixed")
    return loop


def flow_verdict(delta, ci, pass_max, pass_hi, fail_min):
    if not (len(delta) == len(ci) == len(pass_max) == len(pass_hi) == len(fail_min) == 3):
        raise SystemExit("flow verdict expects three metrics")
    passed = all(float(value) <= float(limit) for value, limit in zip(delta, pass_max)) and all(
        float(pair[1]) < float(limit) for pair, limit in zip(ci, pass_hi))
    failed = all(float(value) >= float(limit) for value, limit in zip(delta, fail_min)) and all(
        float(pair[0]) > 0 for pair in ci)
    if passed and failed:
        raise SystemExit("flow verdict overlaps")
    if passed:
        return "PASS"
    if failed:
        return "FAIL"
    return "UNIDENTIFIED"


def first_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip().strip("`*")
    return ""


def fmt(values):
    return " / ".join(f"{float(value):+.3f}" for value in values)


def fetch_once(loop):
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
        raise NotReady(proc.stderr.strip() or "result is not ready")
    measured = json.loads(proc.stdout)
    if measured["status"] != "COMPLETED" or measured["updates"] != 500:
        raise NotReady("result is not a completed 500-update run")
    if measured["matched_clip"] is not True or measured["flow_only"] is not True:
        raise SystemExit("flow-only recipe switches are not the authorized pair")
    if measured["flow_n"] != 500 or measured["trunk_n"] != 0:
        raise SystemExit("flow-only clip logs are not the authorized shape")
    return measured


def fetch(loop):
    deadline = time.time() + int(loop["wait"].get("poll_seconds") or 0)
    while True:
        try:
            return fetch_once(loop)
        except NotReady as exc:
            if time.time() >= deadline:
                raise SystemExit(str(exc))
            time.sleep(30)


def packet(loop, measured, token, sha):
    ci = ", ".join(fmt(pair) for pair in measured["ci95_mm"])
    name = Path(loop["wait"]["directory"]).name
    return "\n".join([
        "仓库 https://github.com/hitsz-oyx/Ref2Dex-agent",
        "提交 " + sha,
        "同一条对话。这是脚本从结果文件提取的 flow-only 臂，不是新实验。",
        "",
        "diagnosis/" + name + " 已完成 500 updates。matched_clip 开，flow_only 开。原点 [-2,-2,-2]，体素 0.005。",
        "means " + fmt(measured["means_mm"]) + " mm。",
        "相对 local3000 " + fmt(measured["delta_mm"]) + " mm。",
        "母序列 CI95 " + ci + "。",
        "脚本判定 " + token + "。",
        "PASS：三项都 ≤ +1.218 / +4.706 / +0.570 mm，且 CI95 上界都 < +1.827 / +7.059 / +0.855 mm。",
        "FAIL：三项都 ≥ +1.827 / +7.059 / +0.855 mm，且 CI95 下界都 > 0。",
        "其余是 UNIDENTIFIED。clip rate 不是结果门。",
        "flow_future 的 pre-clip 范数大于 1：" + f"{measured['flow_gt1']}/500。主干组为空。",
        "",
        "第一行只能是 PASS、FAIL 或 UNIDENTIFIED。",
        "不要提出下一刀。",
        "",
    ])


def measured_block(loop, measured, token):
    ci = ", ".join(fmt(pair) for pair in measured["ci95_mm"])
    return "\n".join([
        "MEASURED",
        "directory: diagnosis/" + Path(loop["wait"]["directory"]).name,
        "updates: 500",
        "means_mm: " + fmt(measured["means_mm"]),
        "delta_vs_local3000_mm: " + fmt(measured["delta_mm"]),
        "ci95_mm: " + ci,
        "script_token: " + token,
        "flow_preclip_gt1: %d/500" % measured["flow_gt1"],
        "trunk_group: empty",
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


def cycles():
    if not STATE.exists():
        return {}
    raw = json.loads(STATE.read_text(encoding="utf-8"))
    if "cycles" in raw:
        return raw["cycles"]
    if raw.get("directory"):
        return {"c2-matched-control": raw}
    return {}


def save_cycles(items):
    STATE.write_text(json.dumps({"cycles": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    loop = load_loop()
    if not loop["enabled"]:
        raise SystemExit("loop is disabled")
    state = cycles()
    current = state.get(loop["cycle"]) or {}
    if current.get("phase") == "codex_done":
        print(json.dumps({"action": "already_done", "token": current.get("token")}, ensure_ascii=False))
        return
    measured = fetch(loop)
    token = flow_verdict(
        measured["delta_mm"], measured["ci95_mm"], loop["pass_max_mm"],
        loop["pass_ci_hi_mm"], loop["fail_min_mm"])
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
    if current.get("phase") == "gpt_accepted":
        reply_token = current["token"]
    else:
        reply = send_gpt(loop, body)
        reply_token = first_line(reply)
        report["sent"] = True
        report["reply_token"] = reply_token
        if reply_token != token:
            notify("讨论循环停住。脚本判定 %s，GPT 第一行是 %s。\n%s" % (token, reply_token or "空", reply[:500]))
            print(json.dumps(report, ensure_ascii=False))
            raise SystemExit("gpt reply is not the script token")
        current = {"phase": "gpt_accepted", "directory": loop["wait"]["directory"], "token": token}
        state[loop["cycle"]] = current
        save_cycles(state)
    prompt = (ROOT / loop["prompts"][reply_token]).read_text(encoding="utf-8")
    prompt += "\n" + measured_block(loop, measured, reply_token)
    resume_codex(loop, prompt)
    state[loop["cycle"]] = {"phase": "codex_done", "directory": loop["wait"]["directory"], "token": token}
    save_cycles(state)
    report["codex"] = "CARD_UPDATED"
    report["commit"] = publish(token)
    print(json.dumps(report, ensure_ascii=False))


def publish(token):
    allowed = {
        "docs/context/recent.md",
        "docs/context/loop-state.json",
        "src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261024-c2-matched-clip.md",
    }
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    dirty = []
    for line in status.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if path not in allowed:
            notify("讨论循环写完但有计划外文件，没有提交。判定 %s。\n%s" % (token, status[:500]))
            return "not_committed"
        dirty.append(path)
    if not dirty:
        notify("讨论循环写完，没有文件变化。判定 %s。" % token)
        return "unchanged"
    subprocess.check_call(["git", "add", "--", *dirty], cwd=ROOT)
    subprocess.check_call(["git", "commit", "-m", "Record the matched-clip flow-only verdict."], cwd=ROOT)
    subprocess.check_call(["git", "push", "origin", "HEAD"], cwd=ROOT)
    subprocess.check_call([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "4-4090_tailscale",
        "cd /home/wbcd/workspace/oyx_ws/Ref2Dex-agent && git fetch origin && git merge --ff-only origin/main",
    ])
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    notify("讨论循环写完。判定 %s。提交 %s。" % (token, sha))
    return sha


if __name__ == "__main__":
    main()
