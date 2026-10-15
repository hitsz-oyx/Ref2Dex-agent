"""Viser browser review of saved actual rollouts, reference poses and contacts."""
import argparse
import json
from pathlib import Path
import os
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser

from rollout_review_data import Corpus, ReviewJournal, visual_meshes, xyzw_to_wxyz


ROOT = Path(__file__).resolve().parents[5]
ASSETS = Path('/home2/wyy/oyx_ws/dexplore/dexplore/data/assets')
VERDICTS = {'可用': 'usable', '不可用': 'unusable', '待定': 'uncertain'}
REASONS = {'无明显问题': 'none', '未抓住': 'missed_grasp', '提前掉落': 'early_drop',
           '手部漂移': 'hand_drift', '疑似穿透': 'suspected_penetration',
           '跟踪偏离': 'tracking_error', '其他': 'other'}
# Reference has the native 16 key bodies; do not invent missing link poses.
EDGES = np.array([(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                  (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                  (0, 13), (13, 14), (14, 15)])


class Viewer:
    def __init__(self, corpus, journal, assets, port, initial=None):
        self.corpus, self.journal, self.assets = corpus, journal, assets
        self.lock = threading.RLock()
        self.playing = False
        self.loaded = False
        self.names = sorted(corpus.by_name)
        self.server = viser.ViserServer(host='127.0.0.1', port=port, label='DExplore 数据审阅')
        if self.server.get_port() != port:
            self.server.stop()
            raise RuntimeError('Requested port unavailable; use another --port')
        s = self.server
        s.scene.set_up_direction('+z')
        s.gui.configure_theme(control_width='large', show_share_button=False)
        s.gui.add_markdown('**真实执行状态回放** · 蓝色手 / 橙色物体；绿色参考；红点为实测接触。\n\n'
                           '旋转、缩放观察后标记片段；标签单独保存。多浏览器共享播放状态。')
        self.selector = s.gui.add_dropdown('序列（全部 660 条）', self.names, initial_value=initial or self.names[0])
        s.gui.add_button('上一条').on_click(lambda _: self.navigate(-1))
        s.gui.add_button('下一条').on_click(lambda _: self.navigate(1))
        self.play = s.gui.add_button('播放 / 暂停')
        self.frame = s.gui.add_slider('帧', 0, 1, 1, 0)
        self.speed = s.gui.add_dropdown('播放速度', ['0.25', '0.5', '1', '2'], initial_value='1')
        self.reference = s.gui.add_checkbox('显示参考', True)
        self.contacts = s.gui.add_checkbox('显示接触点', True)
        self.info = s.gui.add_markdown('载入中')
        s.gui.add_button('相机对准手物').on_click(lambda _: self.center_camera())
        with s.gui.add_folder('人工标注（首尾帧均包含）'):
            self.start = s.gui.add_number('起始帧', 0, min=0, max=1, step=1)
            self.end = s.gui.add_number('结束帧', 0, min=0, max=1, step=1)
            s.gui.add_button('当前帧 → 起始').on_click(lambda _: self.mark('start'))
            s.gui.add_button('当前帧 → 结束').on_click(lambda _: self.mark('end'))
            s.gui.add_button('选择整条').on_click(lambda _: self.mark('all'))
            self.verdict = s.gui.add_dropdown('判定', list(VERDICTS), initial_value='待定')
            self.reason = s.gui.add_dropdown('原因', list(REASONS), initial_value='其他')
            self.note = s.gui.add_text('备注', '', multiline=True)
            s.gui.add_button('保存标注', color='green').on_click(lambda _: self.save())
            self.feedback = s.gui.add_markdown('尚未保存新标注。')
        self.hand = []
        meshes = visual_meshes(assets / 'inspire_hand_new/inspire_hand_right.urdf')
        for index, name in enumerate(corpus.contract['body_names']):
            if name in meshes:
                self.hand.append((index, self.mesh('/actual/hand/' + name, meshes[name], (100, 165, 225))))
        self.table = self.mesh('/table', trimesh.util.concatenate(list(visual_meshes(assets / 'mjcf/table.urdf').values())), (175, 157, 126))
        self.table.visible = False
        self.ref_points = s.scene.add_point_cloud('/reference/keypoints', np.zeros((16, 3)), (70, 220, 100), point_size=.007)
        self.ref_lines = s.scene.add_line_segments('/reference/skeleton', np.zeros((15, 2, 3)), (70, 220, 100), line_width=2)
        self.contact_points = s.scene.add_point_cloud('/actual/contact', np.zeros((5, 3)), np.zeros((5, 3), dtype=np.uint8), point_size=.012)
        self.object = self.ref_object = None
        self.selector.on_update(self.select)
        self.frame.on_update(self.scrub)
        self.play.on_click(self.toggle_play)
        self.reference.on_update(lambda _: self.refresh())
        self.contacts.on_update(lambda _: self.refresh())
        self.load(self.selector.value)

    def mesh(self, name, mesh, color, **kwargs):
        return self.server.scene.add_mesh_simple(name, np.asarray(mesh.vertices, np.float32),
            np.asarray(mesh.faces, np.uint32), color=color, **kwargs)

    def load(self, name):
        with self.lock:
            self.loaded = False
            self.playing = False
            self.record, self.data = self.corpus.load(name)
            self.name = name
            mesh = trimesh.util.concatenate(list(visual_meshes(self.assets / ('mjcf/' + self.record['object'] + '.urdf')).values()))
            for old in (self.object, self.ref_object):
                if old is not None:
                    old.remove()
            self.object = self.mesh('/actual/object', mesh, (238, 155, 50))
            self.ref_object = self.mesh('/reference/object', mesh, (70, 220, 100), wireframe=True, opacity=.4)
            try:
                table = self.corpus.table_pose(name)
                self.table.position = table[:3]
                self.table.wxyz = xyzw_to_wxyz(table[3:7])
                self.table.visible = True
            except (OSError, ValueError) as error:
                self.table.visible = False
                print('Table unavailable:', name, str(error), flush=True)
            maximum = self.record['frames'] - 1
            self.frame.max = self.start.max = self.end.max = maximum
            self.frame.value = self.start.value = 0
            self.end.value = maximum
            self.note.value = ''
            self.verdict.value = '待定'
            self.reason.value = '其他'
            self.feedback.content = f'本序列已有 {self.journal.count(name)} 条标注。'
            self.tick = 0
            self.loaded = True
            self.refresh()
            self.center_camera()

    def select(self, event):
        if event.client is not None:
            self.load(event.target.value)

    def navigate(self, direction):
        with self.lock:
            index = (self.names.index(self.name) + direction) % len(self.names)
            self.selector.value = self.names[index]
            self.load(self.names[index])

    def scrub(self, event):
        if event.client is not None:
            with self.lock:
                self.playing = False
                self.tick = int(event.target.value)
                self.refresh()

    def toggle_play(self, _):
        with self.lock:
            self.playing = not self.playing
            if self.playing and self.tick == self.record['frames'] - 1:
                self.tick = 0
                self.refresh()

    def center_camera(self):
        with self.lock:
            center = self.data['obj'][self.tick, :3, 3].astype(float)
            position = center + [.55, -.65, .4]
            self.server.initial_camera.position = position
            self.server.initial_camera.look_at = center
            for client in self.server.get_clients().values():
                client.camera.position = position
                client.camera.look_at = center

    def refresh(self):
        with self.lock:
            if not self.loaded:
                return
            t, d = self.tick, self.data
            with self.server.atomic():
                self.frame.value = t
                for index, handle in self.hand:
                    handle.position = d['body_state'][t, index, :3]
                    handle.wxyz = xyzw_to_wxyz(d['body_state'][t, index, 3:7])
                for handle, poses in ((self.object, d['obj']), (self.ref_object, d['reference_obj'])):
                    handle.position = poses[t, :3, 3]
                    handle.wxyz = xyzw_to_wxyz(Rotation.from_matrix(poses[t, :3, :3]).as_quat())
                self.ref_points.points = d['reference_hand'][t]
                self.ref_lines.points = d['reference_hand'][t, EDGES]
                for handle in (self.ref_points, self.ref_lines, self.ref_object):
                    handle.visible = self.reference.value
                self.contact_points.points = d['body_state'][t, self.corpus.contract['contact_body_ids'], :3]
                self.contact_points.colors = np.where(d['contact'][t, :, None], [245, 45, 45], [125, 125, 125]).astype(np.uint8)
                self.contact_points.visible = self.contacts.value
                admitted = t in self.record.get('admitted_ticks', [])
                gate = ('通过' if admitted else '未通过') if t < self.record['frames'] - 24 else '末尾无完整24步窗口'
                self.info.content = (f'**{self.name}** · {self.record["split"]}\n\n'
                    f'帧 {t}/{self.record["frames"]-1} · {t/self.corpus.fps:.2f}s · '
                    f'{"播放中" if self.playing else "暂停"}\n\n'
                    f'物体位置误差 **{1000*d["object_error"][t]:.1f} mm**；'
                    f'16点平均手误差 **{1000*d["hand_error"][t]:.1f} mm**\n\n'
                    f'当前实测接触 {int(d["contact"][t].sum())}/5；'
                    f'参考接触 {int((d["reference_contact"][t]>0).sum())}/5\n\n'
                    f'现有物体位置窗口筛选：{gate}。该筛选不保证抓取成功。')

    def mark(self, which):
        with self.lock:
            self.playing = False
            if which == 'all':
                self.start.value, self.end.value = 0, self.record['frames'] - 1
            else:
                getattr(self, which).value = self.tick
            self.refresh()

    def save(self):
        with self.lock:
            self.playing = False
            try:
                path = self.journal.save(self.name, self.start.value, self.end.value,
                    VERDICTS[self.verdict.value], REASONS[self.reason.value], self.note.value)
                self.feedback.content = f'已保存：**{self.name}** 帧 {int(self.start.value)}–{int(self.end.value)}，{self.verdict.value}。\n\n`{path.name}`'
            except (OSError, ValueError) as error:
                self.feedback.content = '**保存失败：** ' + str(error)
            self.refresh()

    def run(self, hours):
        print(json.dumps(dict(status='READY', port=self.server.get_port(), sequences=len(self.names),
                              pid=os.getpid(), output=str(self.journal.output))), flush=True)
        deadline = time.monotonic() + hours * 3600
        last = time.monotonic()
        carry = 0.
        try:
            while time.monotonic() < deadline:
                time.sleep(1 / 30)
                now = time.monotonic()
                with self.lock:
                    if self.playing and self.server.get_clients():
                        carry += (now - last) * self.corpus.fps * float(self.speed.value)
                        advance = int(carry)
                        if advance:
                            carry -= advance
                            self.tick = min(self.tick + advance, self.record['frames'] - 1)
                            if self.tick == self.record['frames'] - 1:
                                self.playing = False
                            self.refresh()
                    else:
                        carry = 0.
                last = now
        finally:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--sequence')
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT / 'outputs' not in output.parents or not 0 < args.hours <= 24:
        raise ValueError('Local outputs directory and 0 < hours <= 24 required')
    corpus = Corpus(args.corpus)
    viewer = Viewer(corpus, ReviewJournal(output, corpus), args.assets, args.port, args.sequence)
    viewer.run(args.hours)


if __name__ == '__main__':
    main()
