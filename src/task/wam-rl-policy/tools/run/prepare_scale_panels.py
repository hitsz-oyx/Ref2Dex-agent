"""Freeze outcome-independent native object panels using existing read-only assets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[5]
PIN='6c9e710b04769bf5d9765e3233820101cb3d718f'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists() or ROOT/'outputs/wam-rl-policy' not in out.parents:raise ValueError('Fresh Task output required')
    meta=json.loads((a.cache/'manifest.json').read_text());assert meta['status']=='COMPLETED'
    objects={}
    for record in meta['records']:
        key=record['object_code']
        if key in objects:
            assert objects[key]['split']==record['split'] and objects[key]['scale']==record['scale']
            continue
        episode=Path(record['raw_episode']);batch=episode.parent.parent
        if (batch/'manifest.json').exists():asset=Path(json.loads((batch/'manifest.json').read_text())['asset_root'])
        else:asset=Path(json.loads((episode.parent/'result.json').read_text())['asset_root'])
        objects[key]=dict(object_code=key,scale=record['scale'],split=record['split'],official_split='TRAIN',source_asset_root=str(asset))
    out.mkdir(parents=True)
    def balanced(rows,size,seed):
        rng=np.random.default_rng(seed);groups=defaultdict(list)
        for row in sorted(rows,key=lambda x:x['object_code']):groups[row['object_code'].split('/')[1].rsplit('-',1)[0]].append(row)
        for values in groups.values():rng.shuffle(values)
        keys=sorted(groups);rng.shuffle(keys);selected=[]
        while len(selected)<size:
            for key in keys:
                if groups[key]:selected.append(groups[key].pop())
                if len(selected)==size:break
        return selected
    train=balanced([r for r in objects.values() if r['split']=='train'],128,243)
    dev=sorted([r for r in objects.values() if r['split']=='dev'],key=lambda x:x['object_code'])
    core=balanced(dev,64,248)
    def panel(name,rows):
        manifests=[]
        for offset in range(0,len(rows),32):
            shard=rows[offset:offset+32];folder=out/('%s-%02d'%(name,offset//32));assets=folder/'assets';assets.mkdir(parents=True)
            first=Path(shard[0]['source_asset_root'])
            for key in ('mjcf','textures'):(assets/key).symlink_to((first/key).resolve(),target_is_directory=True)
            priors={};sources={};closure=set()
            for row in shard:
                code=row['object_code'];source=Path(row['source_asset_root']);scale=row['scale']
                if source not in sources:sources[source]=np.load(source/'datasetv4.1_posedata.npy',allow_pickle=True).item()
                priors[code]={scale:sources[source][code][scale]}
                row['initialization_priors']=len(priors[code][scale]['object_init_z'])
                for key in ('meshdatav3_scaled','meshdatav3_pc_feat'):
                    dest=assets/key/code;dest.parent.mkdir(parents=True,exist_ok=True);target=(source/key/code).resolve()
                    if not target.is_dir():raise FileNotFoundError(target)
                    dest.symlink_to(target,target_is_directory=True)
                    closure.update(q.resolve() for q in target.rglob('*') if q.is_file())
                urdf=assets/'meshdatav3_scaled'/code/'coacd'/('coacd_%03d.urdf'%round(scale*100))
                if not urdf.exists():raise FileNotFoundError(urdf)
            for key in ('mjcf','textures'):closure.update(q.resolve() for q in (first/key).resolve().rglob('*') if q.is_file())
            np.save(assets/'datasetv4.1_posedata.npy',priors,allow_pickle=True)
            size=sum(q.stat().st_size for q in closure)+(assets/'datasetv4.1_posedata.npy').stat().st_size
            if size>512*2**20:raise RuntimeError('Explicit per-shard asset closure exceeded512MiB')
            packet=dict(status='COMPLETE',upstream_commit=PIN,objects=shard,asset_root=str(assets),
                selection='preselected by category/object identity, no model outcomes',real_asset_closure_bytes=size,
                linked_sources_read_only=True,training_allowed=name=='train')
            path=folder/'manifest.json';path.write_text(json.dumps(packet,indent=2)+'\n');manifests.append(str(path))
        return manifests
    manifest=dict(status='COMPLETED',schema='ref2dex.shadow-scale-panels.v1',train_objects=128,dev_objects=len(dev),core_dev_objects=64,
        starts_per_object=4,source=str(a.cache.resolve()),selection_seeds=dict(train=243,core_dev=248),
        train=panel('train',train),dev=panel('dev',dev),core_dev=panel('core-dev',core),
        scope='project DEV; all official TRAIN assets; not official test-unseen')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))


if __name__=='__main__':main()
