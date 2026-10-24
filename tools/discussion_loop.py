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
        "pass_ci_hi_mm", "fail_min_mm", "prompts", "next",
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
    if set(loop["next"]["allow"]) != {"CLOSE_NO_ROUTE"}:
        raise SystemExit("next instruction allow list is fixed")
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


def next_packet(loop, sha):
    allowed = "、".join(loop["next"]["allow"])
    return "\n".join([
        "仓库 https://github.com/hitsz-oyx/Ref2Dex-agent",
        "提交 " + sha,
        "同一条对话。结果已经记上卡，这一问是要下一条指令。",
        "",
        "对照臂 C2-matched-control 复现成立：相对 local3000 +2.406 / +9.333 / +1.139 mm。",
        "flow-only 臂 C2-matched-flow 完成 500 updates。相对 local3000 +1.862 / +6.942 / +1.065 mm。",
        "CI95 下界 +1.604 / +5.763 / +0.880，上界 +2.123 / +8.249 / +1.277。",
        "脚本判定 UNIDENTIFIED，你的第一行也是 UNIDENTIFIED。没有路线结论。",
        "PASS 需要三项都 ≤ +1.218 / +4.706 / +0.570 且上界都 < +1.827 / +7.059 / +0.855。",
        "FAIL 需要三项都 ≥ +1.827 / +7.059 / +0.855 且下界都 > 0。中间项 +6.942 低于 +7.059。",
        "不要改原点，不要重跑 C2-flow-r2，不要重跑 C4-source1，不要靠降学习率处理裁剪率。",
        "",
        "第一行只写一个指令词。脚本目前只能执行：" + allowed + "。",
        "CLOSE_NO_ROUTE 表示接受未识别，停在没有路线结论，不再开新臂。",
        "第一行如果不是这个词，脚本不执行，会把你的全文交回。",
        "",
    ])


def ask_next(loop, state, current, execute):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    body = next_packet(loop, sha)
    report = {"action": "ask_next", "packet_chars": len(body), "sent": False}
    if not execute:
        print(json.dumps(report, ensure_ascii=False))
        return
    if current.get("phase") == "next_pending":
        reply = (ROOT / "tmp" / "loop-next-reply.txt").read_text(encoding="utf-8")
    else:
        reply = send_gpt(loop, body)
        (ROOT / "tmp").mkdir(exist_ok=True)
        (ROOT / "tmp" / "loop-next-reply.txt").write_text(reply, encoding="utf-8")
        report["sent"] = True
    reply_token = first_line(reply)
    report["reply_token"] = reply_token
    prompt_path = loop["next"]["allow"].get(reply_token)
    if not prompt_path:
        kept = ROOT / "docs" / "context" / "gpt-next-reply.md"
        kept.write_text(reply.strip() + "\n", encoding="utf-8")
        current.update(phase="next_unmatched", reply_token=reply_token or "空")
        state[loop["cycle"]] = current
        save_cycles(state)
        notify("讨论循环停住。下一条指令不在允许列表里，没有执行。\n" + reply[:800])
        report["commit"] = publish("next instruction not executed", extra=["docs/context/gpt-next-reply.md"])
        print(json.dumps(report, ensure_ascii=False))
        return
    current.update(phase="next_pending", reply_token=reply_token)
    state[loop["cycle"]] = current
    save_cycles(state)
    prompt = (ROOT / prompt_path).read_text(encoding="utf-8")
    resume_codex(loop, prompt + "\nGPT_REPLY\n" + reply.strip() + "\n")
    current.update(phase="next_done")
    state[loop["cycle"]] = current
    save_cycles(state)
    report["codex"] = "CARD_UPDATED"
    report["commit"] = publish("Record GPT closing the matched-clip claim with no route.")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    loop = load_loop()
    if not loop["enabled"]:
        raise SystemExit("loop is disabled")
    state = cycles()
    current = state.get(loop["cycle"]) or {}
    if current.get("phase") in ("codex_done", "next_pending"):
        ask_next(loop, state, current, args.execute)
        return
    if current.get("phase") in ("next_done", "next_unmatched"):
        print(json.dumps({"action": "already_done", "phase": current.get("phase"), "reply_token": current.get("reply_token")}, ensure_ascii=False))
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
    report["commit"] = publish("Record the matched-clip flow-only verdict.")
    print(json.dumps(report, ensure_ascii=False))


def publish(message, extra=()):
    allowed = {
        "docs/context/recent.md",
        "docs/context/loop-state.json",
        "src/task/ptv3-shared-wam-v6/docs/experiments/probes/P-20261024-c2-matched-clip.md",
        *extra,
    }
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    dirty = []
    for line in status.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if path not in allowed:
            notify("讨论循环写完但有计划外文件，没有提交。%s\n%s" % (message, status[:500]))
            return "not_committed"
        dirty.append(path)
    if not dirty:
        notify("讨论循环写完，没有文件变化。%s" % message)
        return "unchanged"
    subprocess.check_call(["git", "add", "--", *dirty], cwd=ROOT)
    subprocess.check_call(["git", "commit", "-m", message], cwd=ROOT)
    subprocess.check_call(["git", "push", "origin", "HEAD"], cwd=ROOT)
    subprocess.check_call([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "4-4090_tailscale",
        "cd /home/wbcd/workspace/oyx_ws/Ref2Dex-agent && git fetch origin && git merge --ff-only origin/main",
    ])
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    notify("讨论循环写完。%s 提交 %s。" % (message, sha))
    return sha


if __name__ == "__main__":
    main()
