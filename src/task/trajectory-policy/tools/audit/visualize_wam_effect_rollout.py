"""Measured-state reconstruction and fixed grasp curves; never camera footage."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    root=Path(__file__).resolve().parents[5]
    with np.load(args.native/'trajectory.npz') as f:
        data={key:f[key] for key in ('hand','obj','surface_gap','support_gap','table_footprint')}
    with np.load(root/'outputs/consequence-evaluator/old-utility-pw-20261009-r2/canonical.npz') as f:
        points=f['points'][::3]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    time=np.arange(len(data['hand']))/30.
    height=data['obj'][:,:,2,3]-data['obj'][0,:,2,3]
    held=(data['surface_gap']<=.01)&(height>=.03)&~(data['table_footprint'].astype(bool)&(np.abs(data['support_gap'])<=.02))
    held[0]=False
    native_manifest=json.loads((args.native/'manifest.json').read_text())
    adapter=native_manifest.get('control_identity',{}).get('applied',False)
    names=['Teacher','GT tau + Chunk R',('Robot-context WAM' if adapter else 'C final WAM')+' + Chunk R']
    colors=['tab:blue','tab:green','tab:red']
    figure,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for env,name in enumerate(names):
        axes[0].plot(time,height[:,env]*1000,label=name,color=colors[env])
        axes[1].plot(time,data['surface_gap'][:,env]*1000,label=name,color=colors[env])
        axes[2].plot(time,np.linalg.norm(data['hand'][:,env,0]-data['hand'][:,0,0],axis=-1)*1000,label=name,color=colors[env])
    axes[0].axhline(30,color='gray',linestyle='--');axes[0].set(title='Object height above reset',ylabel='mm')
    axes[1].axhline(10,color='gray',linestyle='--');axes[1].set(title='Measured hand-object surface gap',ylabel='mm')
    axes[1].set_yscale('symlog',linthresh=10)
    axes[2].set(title='Wrist deviation from same-trace teacher',ylabel='mm')
    for axis in axes:axis.set_xlabel('Simulation time (s)');axis.legend(fontsize=7);axis.grid(alpha=.25)
    figure.suptitle('Actual measured rollout; '+names[2])
    figure.savefig(args.output/'outcomes.png',dpi=160);plt.close(figure)
    # Common fixed view of the grasp workspace. If an actor leaves it, retain
    # measured distance and explicitly mark the point outside the view.
    early=np.concatenate([data['hand'][:65,:2].reshape(-1,3),data['obj'][:65,:2,:3,3].reshape(-1,3)])
    low=early.min(0)-.15;high=early.max(0)+.15
    center=(low+high)/2;span=max(high-low);low=center-span/2;high=center+span/2
    figure=plt.figure(figsize=(12,4),constrained_layout=True)
    axes=[figure.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
    edges=[(0,base) for base in (1,3,5,7,9)]+[(base,base+1) for base in (1,3,5,7,9)]
    def draw(tick):
        for env,axis in enumerate(axes):
            axis.clear();axis.set_xlim(low[0],high[0]);axis.set_ylim(low[1],high[1]);axis.set_zlim(low[2],high[2])
            axis.set_box_aspect((1,1,1));axis.view_init(elev=24,azim=-65)
            pose=data['obj'][tick,env];cloud=points@pose[:3,:3].T+pose[:3,3]
            hand=data['hand'][tick,env]
            axis.scatter(*cloud.T,s=3,c='tab:blue',alpha=.7)
            axis.scatter(*hand.T,s=12,c='tab:orange')
            for edge in edges:axis.plot(*hand[list(edge)].T,c='tab:orange',linewidth=1.5)
            visible=bool(((hand[0]>=low)&(hand[0]<=high)).all())
            axis.set_title(names[env]+'\nheight %.0f mm | gap %.0f mm'%(height[tick,env]*1000,data['surface_gap'][tick,env]*1000),fontsize=9)
            axis.text2D(.02,.03,'HELD' if held[tick,env] else 'not held'+(' | wrist outside view' if not visible else ''),transform=axis.transAxes,color=colors[env],fontsize=8)
            axis.set_xlabel('x (m)',fontsize=7);axis.set_ylabel('y (m)',fontsize=7);axis.set_zlabel('z (m)',fontsize=7)
        figure.suptitle('Measured-state reconstruction, not camera footage | tick %d | %.2f s'%(tick,tick/30.),fontsize=10)
    draw(48);figure.savefig(args.output/'contact_tick48.png',dpi=150)
    writer=FFMpegWriter(fps=10,codec='libx264',extra_args=['-crf','24','-pix_fmt','yuv420p'])
    with writer.saving(figure,str(args.output/'measured_rollout.mp4'),dpi=100):
        for tick in range(0,len(time),3):draw(tick);writer.grab_frame()
    plt.close(figure)
    (args.output/'manifest.json').write_text(json.dumps(dict(source=str(args.native/'trajectory.npz'),
        kind='measured_state_reconstruction_not_camera_footage',frames=181,fps=10,dt=1/30,
        sampled_ticks=list(range(0,len(time),3)),fixed_workspace_bounds=[low.tolist(),high.tolist()]),indent=2)+'\n')


if __name__=='__main__':main()
