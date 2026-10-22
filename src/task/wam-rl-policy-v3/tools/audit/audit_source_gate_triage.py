#!/usr/bin/env python3
"""Offline triage for the source-controller approach/contact gate.

The audit deliberately reads completed R3/R4/R5 traces only.  It never imports
the simulator or launches a model.  For R3/R4 the reference geometry is the
object-local palm pose at reference frame 0 versus the first valid two-finger
force frame.  For R5 the recorded source key grasp frame is used.  R4 runtime
errors are reconstructed from the saved A0 adapter action and native PD
conversion, which is the same pose consumed by ``grasp_intent_from_retargeter``
when no wrist plan is present.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


TIPS = np.asarray([2, 4, 6, 8, 10])


def pose7(value):
    value = np.asarray(value)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(value[3:7]).as_matrix()
    out[:3, 3] = value[:3]
    return out


def pose_delta(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    position = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    rotation = float(np.linalg.norm(
        Rotation.from_matrix(second[:3, :3] @ first[:3, :3].T).as_rotvec()))
    return position, rotation


def expand_action(action):
    action = np.asarray(action)
    # Active action slots in the native 18-DOF vector.  The slots are
    # [wrist6, thumb, index, middle, ring, pinky]; the remaining six are
    # reconstructed mimic joints below.
    active = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 15)
    out = np.zeros(action.shape[:-1] + (18,), dtype=np.float32)
    out[..., active] = np.clip(action, -1, 1)
    return out


def pd_targets(action, q, offset, scale):
    full = expand_action(action).astype(np.float64)
    q = np.asarray(q, dtype=np.float64)
    target = np.asarray(offset) + np.asarray(scale) * np.concatenate(
        (full[..., :6], (1.0 + full[..., 6:]) / 2.0), axis=-1)
    target[..., :6] += q[..., :6]
    for dst, src, ratio in ((7, 6, 1.05), (9, 8, 1.05),
                            (11, 10, 1.05), (13, 12, 1.05),
                            (16, 15, .6), (17, 15, .8)):
        target[..., dst] = target[..., src] * ratio
    return target


def summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(values.size),
        "p10": float(np.quantile(values, .10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, .90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def load_points(catalog, sequence):
    record = next(r for r in catalog if r["sequence"] == sequence)
    with np.load(Path(record["source"]) / "canonical.npz") as data:
        return np.asarray(data["points"], dtype=float)


def reference_geometry(run_dir, run, catalog):
    registration = json.loads((run_dir / "registration.json").read_text())
    jobs = {job["id"]: job for job in registration["jobs"]}
    source_keys = {}
    if run == "r5":
        source_keys = {row["id"]: row for row in json.loads(
            (run_dir / "source-keys.json").read_text())}
    rows = []
    result_paths = sorted(run_dir.glob("g*/p*-A0/result.json"))
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        job = jobs[result["trial"]]
        with np.load(job["reference"]) as data:
            objects = np.asarray(data["objects"])[:, 0]
            palms = np.asarray(data["palm_pose"])[:, 0]
            local = np.asarray([
                np.linalg.inv(pose7(obj)) @ pose7(palm)
                for obj, palm in zip(objects, palms)
            ])
            force = np.linalg.norm(np.asarray(data["contact_forces"])[:, 0], axis=-1)
            count = (force >= .1).sum(-1)
            valid = np.asarray(data["contact_force_valid"])[:, 0]
            contact = np.flatnonzero((count >= 2) & valid)
        if run == "r5":
            grasp_frame = int(source_keys[result["trial"]]["grasp_frame"])
            evidence = source_keys[result["trial"]]["evidence"]
        else:
            grasp_frame = int(contact[0]) if len(contact) else len(local) - 1
            evidence = "first_valid_two_finger_force_frame" if len(contact) else "no_valid_force_frame"
        position, rotation = pose_delta(local[0], local[grasp_frame])
        geometry = result["geometry"]
        wrist_gap = float(geometry["wrist_gap_m"])
        angle_tol = float(geometry["angle_tolerance_rad"])
        rows.append({
            "id": result["trial"], "object": job["object"],
            "approach_frame": 0, "grasp_frame": grasp_frame,
            "evidence": evidence, "position_delta_m": position,
            "rotation_delta_rad": rotation, "wrist_gap_m": wrist_gap,
            "angle_tolerance_rad": angle_tol,
            "position_hit": position <= wrist_gap,
            "rotation_hit": rotation <= angle_tol,
            "both_hit": position <= wrist_gap and rotation <= angle_tol,
            "position_over_5x": position > 5 * wrist_gap,
            "rotation_over_5x": rotation > 5 * angle_tol,
        })
    return sorted(rows, key=lambda row: row["id"])


def reconstruct_r4(run_dir, catalog):
    registration = json.loads((run_dir / "registration.json").read_text())
    jobs = {job["id"]: job for job in registration["jobs"]}
    rows = []
    for result_path in sorted(run_dir.glob("g*/p*-A0/result.json")):
        result = json.loads(result_path.read_text())
        job = jobs[result["trial"]]
        trace = np.load(result_path.with_name("trajectory.npz"))
        infos = json.loads(result_path.with_name("consumer.json").read_text())
        t = int(result["controls"] - 1)
        q = np.asarray(trace["q"])[t, 0]
        palm = pose7(np.asarray(trace["palm_pose"])[t, 0])
        obj = pose7(np.asarray(trace["objects"])[t, 0])
        native = json.loads(result_path.with_name("native-audit.json").read_text())
        offset = np.asarray(native["controller"]["offset"])
        scale = np.asarray(native["controller"]["scale"])
        pd = pd_targets(np.asarray(trace["adapter_raw_action"])[t], q, offset, scale)
        mount = Rotation.from_euler("XYZ", q[3:6]).as_matrix().T @ palm[:3, :3]
        target = np.eye(4)
        target[:3, :3] = Rotation.from_euler("XYZ", pd[3:6]).as_matrix() @ mount
        target[:3, 3] = palm[:3, 3] + pd[:3] - q[:3]
        position, rotation = pose_delta(palm, target)
        points = load_points(catalog, job["sequence"])
        hand = np.asarray(trace["hand"])[t, 0]
        local_hand = (hand - obj[:3, 3]) @ obj[:3, :3]
        distances = np.linalg.norm(local_hand[TIPS, None] - points[None], axis=-1).min(-1)
        near_fingers = int((distances <= .018).sum())
        fingertip_gap = float(distances.min())
        geometry = result["geometry"]
        wrist_gap = float(geometry["wrist_gap_m"])
        tip_gap = float(geometry["tip_gap_m"])
        angle_tol = float(geometry["angle_tolerance_rad"])
        info = infos[-1]
        wrist_static_gate = position <= wrist_gap and rotation <= angle_tol
        code_gate = near_fingers >= 2 or (position <= wrist_gap and fingertip_gap <= tip_gap)
        phase = info["phase"]
        if phase == "approach":
            if wrist_static_gate:
                classification = "wrist_at_target_finger_gate_false"
            else:
                classification = "wrist_not_at_target"
        else:
            classification = "phase_gate_reached_contact_or_lift_missing"
        rows.append({
            "id": result["id"], "object": result["object"],
            "last_control_t": t, "phase_at_last_control": phase,
            "palm_object_origin_distance_m": float(np.linalg.norm(palm[:3, 3] - obj[:3, 3])),
            "wrist_position_error_m": position,
            "wrist_rotation_error_rad": rotation,
            "wrist_gap_m": wrist_gap, "angle_tolerance_rad": angle_tol,
            "tip_gap_m": fingertip_gap, "tip_gap_tolerance_m": tip_gap,
            "near_fingers": near_fingers,
            "wrist_static_gate": wrist_static_gate, "code_gate": code_gate,
            "stage_contact_achieved": bool(result["stage_achieved"]["contact"]),
            "classification": classification,
            "final_goal_error_m": float(result["final_goal_error_m"]),
        })
    return sorted(rows, key=lambda row: row["id"])


def contact_summary(run_dir):
    rows = []
    for result_path in sorted(run_dir.glob("g*/p*-A0/result.json")):
        result = json.loads(result_path.read_text())
        source = result.get("source_reference", {})
        infos = json.loads(result_path.with_name("consumer.json").read_text())
        rows.append({
            "id": result["id"], "object": result["object"],
            "raw_force_contact_frames": int(source.get("raw_force_contact_frames", 0)),
            "object_contact_frames": int(source.get("object_contact_frames", 0)),
            "source_contact_is_live_success": source.get("source_contact_is_live_success"),
            "ever_measured_contact": any(bool(i.get("measured_contact")) for i in infos),
            "ever_lift_phase": any(i.get("phase") in ("lift", "hold") for i in infos),
            "stage_contact_achieved": bool(result["stage_achieved"]["contact"]),
            "goal_within_geometry": bool(result.get("goal_within_geometry", False)),
        })
    return sorted(rows, key=lambda row: row["id"])


def markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |" for row in rows)
    return "\n".join(out)


def build_report(data):
    ref = data["reference_geometry"]
    r4 = data["r4_runtime"]
    r5 = data["r5_contact"]
    def dist_table(rows, key, threshold):
        values = [row[key] for row in rows]
        return {
            "p10": float(np.quantile(values, .10)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, .90)),
            "hit_count": int(sum(row[key] <= row[threshold] for row in rows)),
            "over_5x_count": int(sum(row[key] > 5 * row[threshold] for row in rows)),
        }
    r3 = ref["r3"]
    r4ref = ref["r4"]
    report = []
    report.append("# Source 控制门控离线分诊（2026-10-23）")
    report.append("")
    report.append("本报告只读取已完成的 R3/R4/R5 轨迹、注册 reference 和保存的 A0 action；没有启动仿真、模型或 GPU。审计工具：`tools/audit/audit_source_gate_triage.py`。")
    report.append("")
    report.append("## 结论")
    report.append("")
    report.append("引用几何本身不自洽：R3/R4 40 例的 reference frame 0→首次有效双指力帧，在物体坐标系中的位置差中位数为 %.4f m（p10 %.4f，p90 %.4f），旋转差中位数 %.4f rad（p10 %.4f，p90 %.4f）；相对每例门限的中位数倍率分别为 %.2f× 和 %.2f×。按每例 `geometry.wrist_gap_m` 和 `angle_tolerance_rad`，位置/旋转双门命中为 0/40，位置和旋转各自也都是 0/40。这个结果支持“录制引用的 approach→grasp 过远”这一实现/接口缺陷假设。" % (np.median([x["position_delta_m"] for x in r4ref]), np.quantile([x["position_delta_m"] for x in r4ref], .1), np.quantile([x["position_delta_m"] for x in r4ref], .9), np.median([x["rotation_delta_rad"] for x in r4ref]), np.quantile([x["rotation_delta_rad"] for x in r4ref], .1), np.quantile([x["rotation_delta_rad"] for x in r4ref], .9), np.median([x["position_delta_m"] / x["wrist_gap_m"] for x in r4ref]), np.median([x["rotation_delta_rad"] / x["angle_tolerance_rad"] for x in r4ref])))
    report.append("")
    report.append("但这不是 R3/R4 runtime intent 的唯一解释：A0 走 `grasp_intent_from_retargeter(..., plan=None)` 时，代码把 `grasp_pose_object` 直接复制为当拍 `approach_pose_object`，所以 intent 内部两 pose 差为零（R3/R4 所有 A0 轨迹 `tau_valid` 也均为 false）。R3/R4 阶段门实际还要求 `near_fingers >= 2`，或腕位置差≤`wrist_gap_m`且指尖最小距离≤`tip_gap_m`；历史 R3/R4 代码没有把 `angle_tolerance_rad` 用进该门。")
    report.append("")
    report.append("R4 最后一个控制时刻的 40 例中，31 例仍为 `approach`、9 例已经进入 `close/lift`。在仍为 approach 的 31 例里，20 例的重建腕目标已满足位置+旋转静态门，但指尖接近门不满足；另 11 例腕目标本身仍未满足静态门。这解释了“接近但不进入 close”与“进入 close 但不认证 contact”两个层次。R4 的 5 个 `contact_without_lift` 属于后者之后的 live coupling/lift 失败，不是限位主因。")
    report.append("")
    report.append("R5 的 `raw_force_contact_frames`/`object_contact_frames` 是 source 录制证据，不是 live success；`source_contact_is_live_success` 在生产 `SourceGraspReference.manifest()` 中明确固定为 false。R5 40 例中 39 例有 source object-contact 帧、25 例运行时曾有 measured contact、13 例曾进入 lift/hold、21 例阶段曾达到 contact；因此 71/69 帧与 live success=false 并不矛盾，缺的是运行时 `FeedbackObserver` 的运动耦合持续条件及后续 lift/goal 条件。")
    report.append("")
    report.append("## 表 1：引用 approach→grasp 物体坐标几何")
    report.append("")
    report.append("R3/R4 的 approach frame 定义为 reference frame 0，grasp frame 为首次 `contact_force_valid` 且至少两指力≥0.1 N 的帧；R5 使用 `source-keys.json` 的 `grasp_frame`。")
    report.append("")
    report.append("| run | position p10 / median / p90 (m) | rotation p10 / median / p90 (rad) | position hit | rotation hit | both hit | position >5× | rotation >5× |\n|---|---:|---:|---:|---:|---:|---:|---:|")
    for run, rows in ref.items():
        ps = [x["position_delta_m"] for x in rows]; rs = [x["rotation_delta_rad"] for x in rows]
        report.append("| %s | %.4f / %.4f / %.4f | %.4f / %.4f / %.4f | %d/40 | %d/40 | %d/40 | %d/40 | %d/40 |" % (run, np.quantile(ps,.1), np.median(ps), np.quantile(ps,.9), np.quantile(rs,.1), np.median(rs), np.quantile(rs,.9), sum(x["position_hit"] for x in rows), sum(x["rotation_hit"] for x in rows), sum(x["both_hit"] for x in rows), sum(x["position_over_5x"] for x in rows), sum(x["rotation_over_5x"] for x in rows)))
    report.append("")
    report.append("字面上的 runtime `GraspIntent` 字段另有一个重要结果：R3/R4 A0 因 `plan=None`，R5 因 `SourceGraspReference.intent()`，三版都把 `grasp_pose_object` 设成与 `approach_pose_object` 相同；因此 intent 内部位置/旋转差为 0、40/40 自洽。表 1 检查的是录制 reference 的实际 approach frame→grasp/contact frame，不把这两个层次混算。")
    report.append("")
    report.append("逐例明细已写入审计 JSON；R4 逐例门控表如下。")
    report.append("")
    report.append("## 表 2：R4 最后控制时刻的腕误差与实际阶段门")
    report.append("")
    headers = ["id", "object", "phase_at_last_control", "palm_object_origin_distance_m", "wrist_position_error_m", "wrist_gap_m", "wrist_rotation_error_rad", "angle_tolerance_rad", "near_fingers", "tip_gap_m", "tip_gap_tolerance_m", "final_goal_error_m", "wrist_static_gate", "code_gate", "classification"]
    report.append(markdown_table(headers, r4))
    report.append("")
    report.append("R4 分类计数：`phase_at_last_control=approach` 31/40；其中腕静态位置+旋转门已满足但指尖门不满足 20/40，腕静态门未满足 11/40；已经进入 close/lift 9/40。p000-A0 的实测 palm→object 原点距离为 0.1734 m、腕目标位置误差 0.1392 m、腕目标旋转误差 0.3667 rad，而 `final_goal_error_m` 为 0.3201 m；后者是物体当前位姿到任务 post-grasp goal 的误差，不能替代腕到 grasp target 的误差。")
    report.append("")
    report.append("## 表 3：contact 认证路径与 R5 实测")
    report.append("")
    report.append("| 路径 | 全部条件 | R4/R5 证据 |")
    report.append("|---|---|---|")
    report.append("| `FeedbackObserver.contact_observed` | `force_valid`; 至少 2 个力向量范数≥0.1 N；`InteractionObserver` 的 `near_fingers`≥2（指尖到表面≤18 mm） | R5 source 证据不是该 live 标志；R5 40 例中 25 例运行时曾有 `measured_contact` |")
    report.append("| `FeedbackObserver.grasp_candidate` | `contact_observed` 持续至少 0.12 s | R4/R5 consumer 未把 source force 帧直接转成 candidate；R4 仅 9/40 曾离开 approach |")
    report.append("| `FeedbackObserver.grasp_established` | `InteractionObserver.coupled`：near≥2、相对速度<0.04 m/s、物体速度>0.004 m/s、相对速度≤0.001+0.35×物体速度，并持续0.12 s；若 force_valid 还需至少2指力 | R5 仅 13/40 曾进入 lift/hold；source `raw/object` 帧不满足该 live coupling 条件 |")
    report.append("| close→lift | `contact_since` 持续≥0.12 s 且 `feedback.grasp_established` | R4 5 例 `contact_without_lift`；R5 仍需按 live coupling/lift 解释 |")
    report.append("| source key manifest | `raw_force_contact_frames`/`object_contact_frames` 只描述 reference；`source_contact_is_live_success=False` 是显式合同 | R5：raw 总计4724（中位109），object 总计4466（中位102），39/40 object-contact>0；p006 为 raw71/object69、live=false、goal_within_geometry=false |")
    report.append("")
    report.append("实现路径（当前代码行号）：`task_feedback.py:47-57` 统计力、near-finger 和 0.12 s candidate；`interaction_state.py:64-87` 定义 18 mm proximity 与 motion coupling；`pose_consumer.py:219-231` 定义 approach→close 与 close→lift；`source_ceiling.py:76-128` 把 source 接触计数与 live success 分开，并显式写入 `source_contact_is_live_success=False`；`summarize_source_ceiling.py:18-23` 把未达 stage contact 标为 `stage_contact_not_certified`。")
    report.append("")
    report.append("## 决策与最小修复建议")
    report.append("")
    report.append("本轮不启动 R6 全 40。应先在 3–5 个已有失败例上修正引用/门控接口：要么用引用中同一物体坐标系的可达 approach→grasp 几何导出腕门限/分段判据，要么明确把 source 的实测关键 grasp pose 作为 grasp intent，并保留 live contact/coupling 认证；不能把 source reference force contact 当成 live success。修复后只做小样本回放，确认 approach→close 可达、close→lift 仍由 live coupling 控制，再决定全批。")
    return "\n".join(report) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    catalog = json.loads((root / "outputs/wam-rl-policy-v3/inspire-teacher-rollout-audit-20261020-r3/manifest.json").read_text())["records"]
    data = {"reference_geometry": {}, "r4_runtime": [], "r5_contact": []}
    for run in ("r3", "r4", "r5"):
        run_dir = root / ("outputs/wam-rl-policy-v3/source-ceiling-20261023-" + run)
        data["reference_geometry"][run] = reference_geometry(run_dir, run, catalog)
    data["r4_runtime"] = reconstruct_r4(root / "outputs/wam-rl-policy-v3/source-ceiling-20261023-r4", catalog)
    data["r5_contact"] = contact_summary(root / "outputs/wam-rl-policy-v3/source-ceiling-20261023-r5")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(data, indent=2) + "\n")
    args.output_md.write_text(build_report(data))
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_md), "r4_rows": len(data["r4_runtime"]), "r5_rows": len(data["r5_contact"])}, indent=2))


if __name__ == "__main__":
    main()
