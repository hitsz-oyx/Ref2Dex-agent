"""Viser full-sequence nine-source review; source arrays are always read-only."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import signal
import shutil
import sys
import threading
import time
import uuid
import numpy as np
TASK=Path(__file__).resolve().parents[2]
ROOT=TASK.parents[2]
sys.path.insert(0,str(TASK/'src'))
from ptv3_shared_wam_v3.review_data import Catalog,NAMES,transform

COLORS=np.array([[245,160,50],[160,170,185],[175,135,225],[65,190,200],[220,120,170],[170,190,90]],np.uint8)


def finite_max(values):
    values=np.asarray(values)
    return float(values[np.isfinite(values)].max()) if np.isfinite(values).any() else None


class Viewer:
    def __init__(self,catalog,output,host,port,source=4,row=14909,frame=75):
        import viser
        self.catalog,self.output=catalog,Path(output).resolve()
        protected=[catalog.native_root,catalog.main_root]+[Path(s['root']) for s in catalog.main['sources']]
        if any(self.output==p.resolve() or p.resolve() in self.output.parents for p in protected):
            raise ValueError('review annotations must stay outside source datasets')
        (self.output/'annotations').mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.playing=False;self.sequence=None;self.handles={}
        self.server=viser.ViserServer(host=host,port=port,label='九源数据 · 物理质量审阅')
        if self.server.get_port()!=port:
            self.server.stop();raise RuntimeError('requested port occupied; choose another --port')
        s=self.server;s.scene.set_up_direction('+z')
        s.scene.add_frame('/reference/origin',axes_length=.1,axes_radius=.002)
        s.gui.configure_theme(control_width='large',show_share_button=False)
        s.gui.add_markdown('**九源完整序列审阅** · 物理单位：米 / 秒。\n\n'
            '完整 TRAIN/DEV 序列可选，未准入的帧也可回放检查。物体为CAD表面点云，手为真实11点；缺失点不补齐。')
        if catalog.recipe.get('quality_admission'):
            s.gui.add_markdown('**严格运动质量筛查已生效**：准入提示使用清洗后索引，原始完整轨迹仍可回看。')
        self.source=s.gui.add_dropdown('数据源',list(NAMES.values()),initial_value=NAMES[source])
        self.split=s.gui.add_dropdown('划分',['train','dev','all'],initial_value='train')
        self.search=s.gui.add_text('搜索序列 / row / batch','')
        self.page=s.gui.add_number('序列列表页（每页200条）',0,min=0,max=0,step=1)
        self.selected=s.gui.add_dropdown('完整序列',['载入中'])
        self.counts=s.gui.add_markdown('')
        previous=s.gui.add_button('上一条序列');following=s.gui.add_button('下一条序列')
        s.gui.add_button('打开 7.4m 异常样本',color='orange').on_click(lambda _:self.open_outlier())
        self.frame=s.gui.add_slider('原始处理帧（从0开始）',0,1,1,0)
        self.target=s.gui.add_dropdown('检查物体',['0'])
        self.future=s.gui.add_slider('未来对照帧 +k',0,24,1,24)
        s.gui.add_button('播放 / 暂停').on_click(lambda _:self.toggle_play())
        self.range=s.gui.add_dropdown('播放范围',['完整序列','当前 H4+24 窗口'],initial_value='完整序列')
        self.speed=s.gui.add_dropdown('速度',['0.25','0.5','1','2'],initial_value='0.5')
        self.loop=s.gui.add_checkbox('循环播放',True)
        self.coordinates=s.gui.add_dropdown('坐标系',['世界坐标','固定检查帧目标坐标'],initial_value='世界坐标')
        s.gui.add_markdown('固定检查帧坐标不会跟随物体移动；避免移动坐标系掩盖飞离。')
        s.gui.add_button('相机对准当前手物').on_click(lambda _:self.fit('current'))
        s.gui.add_button('相机包含当前与未来').on_click(lambda _:self.fit('window'))
        s.gui.add_button('相机包含完整轨迹').on_click(lambda _:self.fit('all'))
        self.show_hand=s.gui.add_checkbox('显示手点',True)
        self.show_future=s.gui.add_checkbox('显示未来物体 / 手点',True)
        self.show_path=s.gui.add_checkbox('显示整条物体轨迹',True)
        self.point_size=s.gui.add_slider('物体点大小（米）',.0005,.02,.0005,.003)
        self.info=s.gui.add_markdown('')
        with s.gui.add_folder('速度 / 加速度曲线（全序列、原始有效时间间隔）'):
            self.plot=s.gui.add_plotly(self.figure(None),aspect=1.2)
        with s.gui.add_folder('人工质量标记（单独保存，不修改数据）'):
            self.verdict=s.gui.add_dropdown('判定',['可疑','可用','待定'],initial_value='可疑')
            self.annotation_range=s.gui.add_dropdown('标记范围',['当前 H4+24 窗口','整条序列'])
            self.note=s.gui.add_text('问题说明','',multiline=True)
            s.gui.add_button('保存质量标记',color='green').on_click(lambda _:self.annotate())
            self.feedback=s.gui.add_markdown('')
        for control in (self.source,self.split,self.search):
            control.on_update(lambda e:self.refresh_options() if e.client is not None else None)
        self.page.on_update(lambda e:self.refresh_page() if e.client is not None else None)
        self.selected.on_update(lambda e:self.choose_label() if e.client is not None else None)
        self.frame.on_update(lambda e:self.scrub() if e.client is not None else None)
        self.target.on_update(lambda e:self.change_target() if e.client is not None else None)
        self.coordinates.on_update(lambda e:self.change_coordinates() if e.client is not None else None)
        for control in (self.future,self.show_hand,self.show_future,self.show_path,self.point_size):
            control.on_update(lambda e:self.render() if e.client is not None else None)
        previous.on_click(lambda _:self.navigate(-1));following.on_click(lambda _:self.navigate(1))
        self.filtered=[]
        self.refresh_options(initial_row=row,initial_frame=frame)

    def source_id(self):
        return next(k for k,v in NAMES.items() if v==self.source.value)

    def refresh_options(self,initial_row=None,initial_frame=None):
        with self.lock:
            self.playing=False
            self.filtered=self.catalog.select(self.source_id(),self.split.value,self.search.value)
            total=self.catalog.select(self.source_id(),self.split.value)
            self.counts.content=f'本源/划分共 **{len(total):,} 条完整序列、{sum(r.windows for r in total):,} 个准入窗口**；搜索命中 {len(self.filtered):,} 条。'
            self.page.max=max(0,(len(self.filtered)-1)//200)
            selected=next((i for i,r in enumerate(self.filtered) if r.row==initial_row),0)
            self.page.value=selected//200
            self.refresh_page(load=False)
            if not self.filtered:
                self.sequence=None
                for h in self.handles.values():h.visible=False
                self.info.content='此数据源没有该划分的序列。DexYCB当前只有TRAIN；未虚构DEV。'
                self.plot.figure=self.figure(None)
                return
            record=self.filtered[selected];self.selected.value=record.label
            self.load(record,initial_frame)

    def refresh_page(self,load=True):
        with self.lock:
            first=int(self.page.value)*200
            page=self.filtered[first:first+200]
            self.selected.options=[r.label for r in page] or ['无匹配序列']
            if page:
                self.selected.value=page[0].label
                if load:self.load(page[0])

    def choose_label(self):
        with self.lock:
            record=next((r for r in self.filtered if r.label==self.selected.value),None)
            if record:self.load(record)

    def load(self,record,frame=None):
        with self.lock:
            self.playing=False
            try:
                data=self.catalog.load(record)
            except Exception as error:
                self.info.content=f'载入失败：{error}'
                raise
            self.sequence=data
            self.frame.max=data.frames-1
            self.frame.value=int(np.clip(frame if frame is not None else (data.ticks[0] if len(data.ticks) else 3),0,data.frames-1))
            self.target.options=[f'{i}: {name}' for i,name in enumerate(data.objects)]
            self.target.value=self.target.options[data.target]
            self.anchor_frame=int(self.frame.value)
            self.plot.figure=self.figure(data)
            self.render(update_path=True);self.fit('window')

    def navigate(self,offset):
        with self.lock:
            if not self.filtered:return
            current=next((i for i,r in enumerate(self.filtered) if r.key==self.sequence.record.key),0) if self.sequence else 0
            index=(current+offset)%len(self.filtered)
            self.page.value=index//200;self.refresh_page(load=False)
            self.selected.value=self.filtered[index].label;self.load(self.filtered[index])

    def open_outlier(self):
        with self.lock:
            self.source.value=NAMES[4];self.split.value='train';self.search.value=''
            self.refresh_options(initial_row=14909,initial_frame=75)

    def target_id(self):return int(self.target.value.split(':',1)[0])

    def anchor(self):
        return None if self.coordinates.value=='世界坐标' else np.linalg.inv(self.sequence.poses[self.anchor_frame,self.target_id()])

    def change_target(self):
        with self.lock:
            self.anchor_frame=int(self.frame.value);self.render(update_path=True);self.fit('window')

    def change_coordinates(self):
        with self.lock:
            if self.sequence is None:return
            self.anchor_frame=int(self.frame.value);self.render(update_path=True);self.fit('window')

    def scrub(self):
        with self.lock:self.playing=False;self.render()

    def points(self,name,points,color,size,visible=True):
        points=np.asarray(points,np.float32).reshape(-1,3)
        exists=len(points)>0
        if not exists:points=np.zeros((1,3),np.float32)
        color=np.broadcast_to(np.asarray(color,np.uint8),(len(points),3)).copy()
        if name not in self.handles:
            self.handles[name]=self.server.scene.add_point_cloud('/review/'+name,points,color,point_size=size,precision='float32',point_shading='flat')
        else:
            self.handles[name].points=points;self.handles[name].colors=color;self.handles[name].point_size=size
        self.handles[name].visible=bool(visible and exists)

    def line(self,name,points,color,width,visible=True):
        points=np.asarray(points,np.float32).reshape(-1,2,3)
        exists=len(points)>0
        if not exists:points=np.zeros((1,2,3),np.float32)
        color=np.broadcast_to(np.asarray(color,np.uint8),(len(points),2,3)).copy()
        if name not in self.handles:
            self.handles[name]=self.server.scene.add_line_segments('/review/'+name,points,color,line_width=width)
        else:
            self.handles[name].points=points;self.handles[name].colors=color
        self.handles[name].visible=bool(visible and exists)

    def render(self,update_path=False):
        with self.lock,self.server.atomic():
            d=self.sequence
            if d is None:return
            t=int(self.frame.value);f=min(t+int(self.future.value),d.frames-1);target=self.target_id();anchor=self.anchor()
            for h in self.handles.values():h.visible=False
            clouds,hands,valid=d.frame(t,anchor);future,fhand,fvalid=d.frame(f,anchor)
            for i,p in enumerate(clouds):
                color=COLORS[0] if i==target else COLORS[(i%5)+1]
                self.points(f'object{i}',p,color,float(self.point_size.value))
                self.points(f'future{i}',future[i],(245,220,100),float(self.point_size.value),self.show_future.value and f!=t)
            for side,color in [(0,(70,155,245)),(1,(85,215,135))]:
                self.points(f'hand{side}',hands[side][valid[side]],color,.008,self.show_hand.value)
                self.points(f'futurehand{side}',fhand[side][fvalid[side]],(190,110,225),.006,self.show_hand.value and self.show_future.value and f!=t)
            if update_path or 'path' not in self.handles:
                self.line('path',d.path_segments(target,anchor),(235,150,40),2,self.show_path.value)
            else:
                self.handles['path'].visible=bool(self.show_path.value)
            local=d.path_segments(target,anchor,max(0,t-3),min(t+25,d.frames))
            self.line('window',local,(255,85,60),4,self.show_path.value)
            info=d.horizon_info(t,target)
            # Explicit 10cm ruler; no per-sample spatial normalization.
            origin=np.zeros(3) if anchor is not None else d.centers[t,target]
            ruler=np.array([[origin,origin+[.1,0,0]]],np.float32)
            self.line('ruler',ruler,(245,245,245),3,True)
            maximum=info['max_query_displacement_m']
            self.info.content=(f'**{NAMES[d.record.source_id]} / {d.record.split}** · row `{d.record.row}`\n\n'
                f'`{d.record.sequence}`\n\n帧 **{t}/{d.frames-1}**，原始时间 **{d.time[t]:.4f}s**；'
                f'当前帧 **{"实测" if info["observed"] else "插值"}**。\n\n'
                f'当前检查物体 `{target}`，未来 {info["seconds"]:.3f}s 最大query位移 **{maximum:.4f}m**。\n\n'
                if maximum is not None else f'帧{t}：当前物体无同一segment的有效未来标签。\n\n')
            self.info.content+=f'当前质心速度 `{info["speed_mps"]:.3f}` m/s，加速度 `{info["acceleration_mps2"]:.2f}` m/s²。\n\n' if info['speed_mps'] is not None and info['acceleration_mps2'] is not None else '当前速度/加速度不可计算（边界、缺失或跨段）。\n\n'
            if d.motion_quality:
                self.info.content+=f'自动筛查：从帧 **{d.motion_quality["first_suspect_frame"]}** 起隔离；类型 `{", ".join(d.motion_quality["reasons"])}`。原始整段仍可回放。\n\n'
            self.info.content+=f'当前anchor是否属于冻结准入索引：**{"是" if info["training_window"] else "否（仅原始轨迹检查）"}**。白线长度 **0.1m**。\n\n文件：`{d.origin}`'

    def fit(self,mode):
        with self.lock:
            d=self.sequence
            if d is None:return
            t=int(self.frame.value);anchor=self.anchor();clouds,hands,valid=d.frame(t,anchor)
            points=[p for p in clouds if len(p)]+[hands[valid]]
            if mode=='window':
                clouds,hands,valid=d.frame(min(t+24,d.frames-1),anchor)
                points += [p for p in clouds if len(p)]+[hands[valid]]
                path=d.path_segments(self.target_id(),anchor,max(0,t-3),min(t+25,d.frames))
                if len(path):points.append(path.reshape(-1,3))
            elif mode=='all':
                path=d.path_segments(self.target_id(),anchor)
                if len(path):points.append(path.reshape(-1,3))
            points=np.concatenate([p for p in points if len(p)])
            center=(points.min(0)+points.max(0))/2;distance=max(float(np.linalg.norm(np.ptp(points,axis=0))),.3)
            position=center+np.array([.8,-1,.6])*distance
            self.server.initial_camera.position=position;self.server.initial_camera.look_at=center
            for client in self.server.get_clients().values():client.camera.position=position;client.camera.look_at=center

    def figure(self,data):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.12)
        if data is not None:
            for i in range(len(data.objects)):
                fig.add_trace(go.Scatter(x=data.time,y=data.speed[:,i],name=f'物体{i}速度',connectgaps=False),row=1,col=1)
                fig.add_trace(go.Scatter(x=data.time,y=data.acceleration[:,i],name=f'物体{i}加速度',connectgaps=False),row=2,col=1)
        fig.update_yaxes(title_text='m/s',row=1,col=1);fig.update_yaxes(title_text='m/s²',row=2,col=1)
        fig.update_xaxes(title_text='原始时间（秒）',row=2,col=1)
        fig.update_layout(margin=dict(l=45,r=5,t=15,b=40),height=420,legend=dict(orientation='h'),template='plotly_dark')
        return fig

    def toggle_play(self):
        with self.lock:
            if self.sequence is None:return
            self.playing=not self.playing
            if self.playing:
                t=int(self.frame.value)
                self.play_bounds=(0,self.sequence.frames-1) if self.range.value=='完整序列' else (max(0,t-3),min(t+24,self.sequence.frames-1))
                if t>=self.play_bounds[1]:self.frame.value=self.play_bounds[0]

    def tick(self):
        with self.lock:
            if not self.playing or self.sequence is None:return
            low,high=self.play_bounds;t=int(self.frame.value)+1
            if t>high:
                if self.loop.value:t=low
                else:self.playing=False;return
            self.frame.value=t;self.render()

    def annotate(self):
        with self.lock:
            if self.sequence is None:return
            d=self.sequence;t=int(self.frame.value)
            low,high=(0,d.frames) if self.annotation_range.value=='整条序列' else (max(0,t-3),min(t+25,d.frames))
            if shutil.disk_usage(self.output).free<20*2**30:
                self.feedback.content='可用空间低于20GiB，暂停新增标记。';return
            if len(self.note.value)>2000:
                self.feedback.content='说明最多2000字。';return
            if sum(1 for _ in (self.output/'annotations').glob('*.json'))>=10000:raise ValueError('annotation budget reached')
            row=dict(schema='ref2dex.nine-source-quality-review.v1',annotation_id=uuid.uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),source_id=d.record.source_id,source=NAMES[d.record.source_id],
                split=d.record.split,row=d.record.row,sequence=d.record.sequence,payload=d.origin,
                frame_start=low,frame_end_exclusive=high,inspection_frame=t,target_object=self.target_id(),
                verdict=self.verdict.value,note=self.note.value,measurements=d.horizon_info(t,self.target_id()),
                recipe=self.catalog.recipe,changes_training_admission=False)
            path=self.output/'annotations'/(row['annotation_id']+'.json')
            with path.open('x') as f:json.dump(row,f,ensure_ascii=False,indent=2)
            self.feedback.content=f'已保存 `{path.name}`；训练数据未修改。'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v3/nine-source-recipe-20261015-r1')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/ptv3-shared-wam-v3/nine-source-review-20261015-r1')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8769)
    p.add_argument('--source',type=int,choices=list(NAMES),default=4)
    p.add_argument('--row',type=int,default=14909);p.add_argument('--frame',type=int,default=75)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--max-seconds',type=int,default=86400)
    a=p.parse_args()
    if not 1<=a.max_seconds<=86400:raise ValueError('viewer deadline must be within24h')
    catalog=Catalog(a.recipe)
    if a.check_only:
        reports=[]
        for sid in NAMES:
            record=catalog.select(sid,'train')[0];sequence=catalog.load(record);t=int(sequence.ticks[0]) if len(sequence.ticks) else 3
            sequence.frame(t)
            reports.append(dict(source_id=sid,train_sequences=len(catalog.select(sid,'train')),
                dev_sequences=len(catalog.select(sid,'dev')),train_windows=sum(r.windows for r in catalog.select(sid,'train')),
                dev_windows=sum(r.windows for r in catalog.select(sid,'dev')),first=sequence.horizon_info(t,sequence.target)))
        print(json.dumps(dict(status='PASS',sources=reports),ensure_ascii=False),flush=True);return
    viewer=Viewer(catalog,a.output,a.host,a.port,a.source,a.row,a.frame)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'server.json').write_text(json.dumps(dict(status='RUNNING',url=f'http://{a.host}:{a.port}',
        pid=__import__('os').getpid(),recipe=str(a.recipe.resolve()),source_id=viewer.source_id(),
        row=viewer.sequence.record.row if viewer.sequence else None),indent=2)+'\n')
    print(f'九源数据可视化：http://127.0.0.1:{a.port}',flush=True)
    stopped=threading.Event();started=time.monotonic()
    signal.signal(signal.SIGTERM,lambda n,f:stopped.set());signal.signal(signal.SIGINT,lambda n,f:stopped.set())
    try:
        while not stopped.is_set() and time.monotonic()-started<a.max_seconds and not (a.output/'STOP').exists():
            viewer.tick();stopped.wait(1/(30*float(viewer.speed.value)))
    finally:
        viewer.server.stop()
        state=json.loads((a.output/'server.json').read_text());state['status']='STOPPED'
        (a.output/'server.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
