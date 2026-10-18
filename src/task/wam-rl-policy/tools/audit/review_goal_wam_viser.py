"""Viser comparison of observed state, scheduled goals and cached real WAM tau."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np

TASK = Path(__file__).resolve().parents[2]
ROOT = TASK.parents[2]
sys.path.insert(0, str(TASK/'src'))
from wam_rl_policy.goal_scheduler import transform_points

COLORS = dict(current=(75, 165, 145), goal=(236, 167, 64),
              prediction=(94, 159, 245), measured=(110, 122, 138), future=(180, 104, 198))
MODE = dict(PREPARE='准备：收短物体目标', TRACK='沿参考推进', RECOVER='局部恢复',
            COMPLETE='参考末端达标', TIMEOUT='超时保持')
STATE = dict(UNKNOWN='手部证据未知', APPROACH='手还远', PREPARE='接触构型未形成',
             CONTACT_CANDIDATE='指尖接近，尚未确认随动', COUPLED='观测到持续随动', LOST='随动证据丢失')


def skeleton(points):
    points = np.asarray(points)
    pairs = [(0, i) for i in range(1, 6)]+[(i, i+5) for i in range(1, 6)]
    return points[np.asarray(pairs)]


def trajectory_segments(points):
    points = np.asarray(points)
    return np.stack([points[:-1], points[1:]], axis=-2).reshape(-1, 2, 3)


class Viewer:
    def __init__(self, directory, host, port):
        import viser
        self.directory = Path(directory)
        self.catalog = json.loads((self.directory/'predictions.json').read_text())
        self.manifest = json.loads((self.directory/'inference-manifest.json').read_text())
        if self.manifest['status'] != 'COMPLETED':
            raise ValueError('completed real inference cache required')
        self.data = []
        for case in self.catalog['cases']:
            with np.load(self.directory/case['file'], allow_pickle=False) as arrays:
                self.data.append({k:arrays[k].copy() for k in arrays.files})
        self.lock = threading.RLock()
        self.handles = {}
        self.playing = False
        self.server = viser.ViserServer(host=host, port=port, label='Goal → WAM τ · 真实预测审阅')
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError('requested port is occupied; no automatic port switch')
        self.server.scene.set_up_direction('+z')
        self.server.gui.configure_theme(control_width='large', show_share_button=False)
        gui = self.server.gui
        gui.add_markdown('## Goal → WAM τ\n\n**绿色**：当前实测物体 · **灰色**：当前实测手\n\n'
            '**金色**：输入物体 goal · **蓝色**：WAM 预测手与物体 query\n\n'
            '固定同一观测，比较三个 goal。没有执行机器人动作；当前观测不会被预测推动。')
        gui.add_markdown(f'冻结 v4 **step {self.manifest["checkpoint_step"]}** · H4@30Hz → 未来24帧 / 0.8s。'
                         '\n\nC 是预测代理，不是接触真值或动作评分。')
        self.selected = gui.add_dropdown('真实记录', [c['name'] for c in self.catalog['cases']])
        self.frame = gui.add_slider('当前观测快照', 0, len(self.data[0]['tick'])-1, 1, 0)
        self.future = gui.add_slider('预测未来帧（当前状态保持不动）', 0, 24, 1, 24)
        self.layout = gui.add_dropdown('场景布局', ['三组 goal 并排', '只看当前调度'])
        gui.add_button('播放 / 暂停预测').on_click(lambda _: self.toggle())
        self.speed = gui.add_dropdown('预测播放速度', ['0.25', '0.5', '1'], initial_value='0.5')
        gui.add_button('下一个观测状态').on_click(lambda _: self.next_state())
        gui.add_button('跳到随动状态').on_click(lambda _: self.jump_coupled())
        gui.add_button('相机对准当前场景').on_click(lambda _: self.fit())
        self.info = gui.add_markdown('')
        with gui.add_folder('显示图层'):
            self.show_goal = gui.add_checkbox('输入物体 goal', True)
            self.show_trails = gui.add_checkbox('WAM 手部预测轨迹', True)
            self.show_scene = gui.add_checkbox('WAM 物体32点预测', True)
            self.show_c = gui.add_checkbox('C：当前表面锚点着色', False)
            self.show_observed = gui.add_checkbox('录制的真实未来（仅审阅对照）', False)
            self.show_reference = gui.add_checkbox('已知完整参考路径', False)
        self.explanation = gui.add_markdown('')
        gui.add_markdown('**读图方式**\n\n1. 选当前观测，三栏共用相同手物历史。\n'
            '2. 金色物体是三个输入 goal，可能受检查点／预算约束而重合。\n'
            '3. 拖动未来帧，观察蓝色手与32个物体点。\n'
            '4. 切换当前观测，调度器才根据记录中的实际进度更新。\n\n'
            '手使用实测／预测的11点骨架；物体未来32点来自模型，不由 goal 插值伪造。'
            '\n\nC 着色固定在当前物体表面锚点，表示对应点的未来接触预测，不是未来表面位置。')
        self.selected.on_update(lambda e: self.choose() if e.client is not None else None)
        for control in (self.frame, self.future, self.show_goal, self.show_trails,
                        self.show_scene, self.show_c, self.show_observed, self.show_reference):
            control.on_update(lambda e: self.render() if e.client is not None else None)
        self.layout.on_update(lambda e: self.change_layout() if e.client is not None else None)
        self.server.on_client_connect(lambda client: self.fit(client))
        self.render()

    def case_index(self):
        return next(i for i, c in enumerate(self.catalog['cases']) if c['name'] == self.selected.value)

    def choose(self):
        with self.lock:
            self.playing = False
            self.frame.max = len(self.data[self.case_index()]['tick'])-1
            self.frame.value = 0
            self.render()
            self.fit()

    def next_state(self):
        with self.lock:
            self.playing = False
            self.frame.value = min(int(self.frame.value)+1, int(self.frame.max))
            self.render()

    def jump_coupled(self):
        with self.lock:
            rows = self.catalog['cases'][self.case_index()]['decisions']
            index = next((i for i, r in enumerate(rows) if r['interaction']['state'] == 'COUPLED'), 0)
            self.frame.value = index
            self.render()
            self.fit()

    def toggle(self):
        with self.lock:
            self.playing = not self.playing
            if self.playing and self.future.value == 24:
                self.future.value = 0

    def change_layout(self):
        self.render()
        self.fit()

    def cloud(self, name, points, colors, size, visible=True):
        points = np.asarray(points, np.float32).reshape(-1, 3)
        colors = np.broadcast_to(np.asarray(colors, np.uint8), points.shape).copy()
        if name not in self.handles:
            self.handles[name] = self.server.scene.add_point_cloud(name, points, colors,
                point_size=size, precision='float32', point_shape='circle', point_shading='flat')
        else:
            self.handles[name].points = points
            self.handles[name].colors = colors
        self.handles[name].visible = bool(visible)

    def lines(self, name, points, color, width, visible=True):
        points = np.asarray(points, np.float32).reshape(-1, 2, 3)
        if not len(points):
            points = np.zeros((1, 2, 3), np.float32)
            visible = False
        colors = np.broadcast_to(np.asarray(color, np.uint8), points.shape).copy()
        if name not in self.handles:
            self.handles[name] = self.server.scene.add_line_segments(name, points, colors, line_width=width)
        else:
            self.handles[name].points = points
            self.handles[name].colors = colors
        self.handles[name].visible = bool(visible)

    def label(self, name, position, text):
        if name not in self.handles:
            self.handles[name] = self.server.scene.add_label(name, text, position=position,
                anchor='bottom-center', font_screen_scale=.8)
        else:
            self.handles[name].position = position
            self.handles[name].text = text
        self.handles[name].visible = True

    def spacing(self):
        d = self.data[self.case_index()]
        return max(.48, 2.5*float(np.linalg.norm(np.ptp(d['surface'], axis=0))))

    def fit(self, client=None):
        with self.lock:
            d = self.data[self.case_index()]
            center = d['object_pose'][int(self.frame.value), :3, 3].copy()
            distance = self.spacing()*(1.65 if self.layout.value == '三组 goal 并排' else .85)
            for target in [client] if client is not None else self.server.get_clients().values():
                target.camera.look_at = center+[0, 0, .03]
                target.camera.position = center+np.array([.2, -distance, distance*.72])
                target.camera.up_direction = (0, 0, 1)

    def render(self):
        with self.lock, self.server.atomic():
            index, f, k = self.case_index(), int(self.frame.value), int(self.future.value)
            case, d = self.catalog['cases'][index], self.data[index]
            row = case['decisions'][f]
            for handle in self.handles.values():
                handle.visible = False
            now = d['object_pose'][f]
            actual = transform_points(now, d['surface'])
            anchors = transform_points(now, d['interaction_points'])
            selected_only = self.layout.value == '只看当前调度'
            for variant, title in enumerate(self.catalog['variants']):
                if selected_only and variant != 1:
                    continue
                shift = np.array([0. if selected_only else (variant-1)*self.spacing(), 0., 0.])
                prefix = f'/goals/{variant}'
                self.cloud(prefix+'/current', actual+shift, COLORS['current'], .003)
                self.cloud(prefix+'/hand', d['hand'][f]+shift, COLORS['measured'], .006)
                self.lines(prefix+'/hand-bones', skeleton(d['hand'][f]+shift), COLORS['measured'], 3)
                self.cloud(prefix+'/goal', transform_points(d['goals'][f, variant], d['surface'])+shift,
                           COLORS['goal'], .002, self.show_goal.value)
                future_hand = d['predicted_hand'][f, variant, k]+shift
                self.cloud(prefix+'/predicted-hand', future_hand, COLORS['prediction'], .006)
                self.lines(prefix+'/predicted-bones', skeleton(future_hand), COLORS['prediction'], 4)
                trails = trajectory_segments(d['predicted_hand'][f, variant, :, [0, 6, 7, 8, 9, 10], :].transpose(1, 0, 2))
                self.lines(prefix+'/hand-trails', trails+shift, COLORS['prediction'], 1.4, self.show_trails.value)
                self.cloud(prefix+'/predicted-object', d['predicted_scene'][f, variant, k]+shift,
                           COLORS['prediction'], .005, self.show_scene.value)
                self.lines(prefix+'/object-flow', np.stack([d['predicted_scene'][f, variant, 0],
                    d['predicted_scene'][f, variant, k]], axis=1)+shift,
                    COLORS['prediction'], 1.2, self.show_scene.value)
                p = d['contact'][f, variant, max(k-1, 0)]
                colors = np.stack([245*p+80*(1-p), 90*p+120*(1-p), 65*p+180*(1-p)], -1)
                mask = d['interaction_valid'][f, variant, max(k-1, 0)]
                colors[~mask] = (130, 130, 130)
                self.cloud(prefix+'/contact', anchors+shift, colors, .005, self.show_c.value and k > 0)
                observed = bool(d['observed_future_valid'][f, k])
                self.cloud(prefix+'/observed-future-hand', d['observed_hand'][f, k]+shift,
                           COLORS['future'], .004, self.show_observed.value and observed)
                self.lines(prefix+'/observed-future-bones', skeleton(d['observed_hand'][f, k])+shift,
                           COLORS['future'], 2, self.show_observed.value and observed)
                self.cloud(prefix+'/observed-future-object', d['observed_scene'][f, k]+shift,
                           COLORS['future'], .003, self.show_observed.value and observed)
                ref = d['reference'][:, :3, 3]+shift
                self.lines(prefix+'/reference', np.stack([ref[:-1], ref[1:]], 1),
                           (160, 180, 170), 1, self.show_reference.value)
                goal_mm = np.linalg.norm(d['goal_delta'][f, variant], axis=-1).mean()*1000
                tau_mm = np.linalg.norm(d['predicted_hand'][f, variant, k]-d['hand'][f], axis=-1).mean()*1000
                self.label(prefix+'/title', now[:3, 3]+shift+[0, 0, .22],
                           f'{title}\ngoal 对应点 {goal_mm:.1f} mm · 手预测位移 {tau_mm:.1f} mm')
            interaction = row['interaction']
            self.info.content = (f'**{case["name"]} · 实测 tick {int(d["tick"][f])} / {len(d["reference"])-1}**\n\n'
                f'当前记录时间 **{d["time"][f]:.2f}s**；蓝色预测未来 **+{k/30:.3f}s**。\n\n'
                f'**{MODE[row["mode"]]}** · {STATE[interaction["state"]]}\n\n'
                f'确认锚点 #{row["confirmed_index"]} → 局部匹配 #{row["projected_index"]} '
                f'→ 调度目标 #{row["target_reference_index"]:.2f}\n\n'
                f'前视 **{row["lookahead_m"]*1000:.1f} mm**；局部误差 **{row["reference_error_m"]*1000:.1f} mm**。\n\n'
                f'选择理由：`{row["reason"]}`')
            goal_difference = np.linalg.norm(d['goal_delta'][f]-d['goal_delta'][f, 1:2], axis=-1).mean(-1)*1000
            hand_difference = np.linalg.norm(d['predicted_hand'][f, :, k]-d['predicted_hand'][f, 1:2, k], axis=-1).mean(-1)*1000
            self.explanation.content = ('**相对当前调度的对照差异**\n\n'
                f'较短／较长 goal 差：{goal_difference[0]:.2f} / {goal_difference[2]:.2f} mm\n\n'
                f'当前预测时刻的手点差：{hand_difference[0]:.2f} / {hand_difference[2]:.2f} mm\n\n'
                '前视受下个检查点和跨度上限约束，三栏可能重合。差异反映条件敏感性，不代表物理可行。\n\n'
                +('当前预测时刻有录制未来可供对照。' if observed else '当前预测时刻超过记录末端，真实未来图层已隐藏。'))
            state = dict(case=case['name'], snapshot=f, source_tick=int(d['tick'][f]),
                         future_frame=k, mode=row['mode'], panels=1 if selected_only else 3)
            (self.directory/'viewer-state.json').write_text(json.dumps(state)+'\n')

    def run(self, hours):
        stop = threading.Event()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        process = dict(pid=os.getpid(), host=self.server.get_host(), port=self.server.get_port(),
            started_utc=datetime.now(timezone.utc).isoformat(), lifetime_hours=hours,
            command=sys.argv, directory=str(self.directory.resolve()),
            stop_condition='SIGTERM/SIGINT or lifetime expires; CPU display only, no CUDA model retained')
        (self.directory/'viewer-process.json').write_text(json.dumps(process, indent=2)+'\n')
        print(json.dumps(dict(status='READY', **process)), flush=True)
        deadline, last, carry = time.monotonic()+hours*3600, time.monotonic(), 0.
        try:
            while time.monotonic() < deadline and not stop.wait(1/30):
                now = time.monotonic()
                with self.lock:
                    if self.playing and self.server.get_clients():
                        carry += (now-last)*30*float(self.speed.value)
                        if carry >= 1:
                            self.future.value = (int(self.future.value)+int(carry)) % 25
                            carry %= 1
                            self.render()
                    else:
                        carry = 0.
                last = now
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.hours <= 24:
        raise ValueError('0 < hours <= 24 required')
    Viewer(args.directory, args.host, args.port).run(args.hours)


if __name__ == '__main__':
    main()
