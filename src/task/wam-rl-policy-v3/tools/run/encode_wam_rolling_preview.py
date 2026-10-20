"""Stream labeled full-sequence WAM24/display8 camera PNGs into videos."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview',type=Path,required=True)
    args=parser.parse_args();out=args.preview.resolve()
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    meta=json.loads((out/'manifest.json').read_text())
    render=json.loads((out/'render-manifest.json').read_text())
    if meta.get('preview_mode')!='rolling_measured_history' or render['status']!='COMPLETED' or render['fk_max_mm']>2:
        raise ValueError('verified rolling camera render required')
    with np.load(out/'preview.npz') as z:
        anchors=z['anchor_tick'].copy();horizons=z['horizon_frame'].copy();valid=z['prediction_valid'].copy()
    n=meta['frames'];font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,18);small=ImageFont.truetype(font_path,16)
    def labeled(i):
        with Image.open(out/'camera-frames'/('%04d.png'%i)) as raw:
            w,h=raw.size;frame=Image.new('RGB',(w,h+150),'#151b24');frame.paste(raw,(0,70))
        draw=ImageDraw.Draw(frame)
        draw.text((16,10),'Measured teacher / native Inspire',font=font,fill='#dde4ec')
        draw.text((w//2+16,10),'WAM24 / show8 / fitted MANO',font=font,fill='#40d4eb')
        draw.text((16,39),'Measured H4 refreshed every8 frames; same recorded object in both lanes',font=small,fill='#a9b7c8')
        if valid[i]:
            offset=int(horizons[i]);anchor=int(anchors[i])
            text='t=%.2fs | tick%d | replan anchor%d | forecast frame%d/8 (+%.3fs)'%(i/30,i,anchor,offset,offset/30)
            draw.text((16,h+80),text,font=small,fill='white')
            draw.text((16,h+104),'Raw WAM vs teacher %.1fmm | MANO fit %.1fmm | %s'%(meta['per_frame_prediction_error_mm'][i],meta['per_frame_fit_mean_mm'][i],
                'NEW PREDICTION' if offset==1 else 'same prediction chunk'),font=small,fill='#f7cc6b')
        else:
            reason='H4 startup: no forecast yet' if i<4 else 'Outside admitted prefix: teacher replay only'
            draw.text((16,h+80),'t=%.2fs | tick%d | %s'%(i/30,i,reason),font=small,fill='#f7cc6b')
        draw.text((16,h+128),'Recorded-observation forecast review; no robot execution. MANO is approximate. No output smoothing.',font=small,fill='#a9b7c8')
        return frame
    selected=[int(valid.nonzero()[0][0]),min(68,n-1),min(180,n-1),min(360,n-1),min(540,n-1),n-1]
    stills=[labeled(i) for i in selected]
    sheet=Image.new('RGB',(stills[0].width,stills[0].height*len(stills)))
    for row,still in enumerate(stills):sheet.paste(still,(0,row*still.height))
    sheet.resize((960,round(sheet.height*960/sheet.width))).save(out/'rolling-contact-sheet.png')
    labeled(min(180,n-1)).save(out/'rolling-middle.png')
    videos=[]
    for repeat,name in [(1,'rolling.mp4'),(2,'rolling-slow2x.mp4')]:
        path=out/name
        if path.exists():raise ValueError('refusing to overwrite video '+name)
        size=stills[0].size
        process=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','%dx%d'%size,
            '-r','30','-i','-','-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
        try:
            for i in range(n):
                payload=np.asarray(labeled(i)).tobytes()
                for _ in range(repeat):process.stdin.write(payload)
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg failed')
        videos.append(dict(file=name,frames=n*repeat,duration_s=n*repeat/30))
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>150*1024**2:raise ValueError('150MiB artifact budget exceeded')
    report=dict(status='COMPLETED',videos=videos,total_bytes=size,frames=n)
    (out/'encode-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
