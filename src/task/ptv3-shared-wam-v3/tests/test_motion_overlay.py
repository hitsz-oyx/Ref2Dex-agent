import copy,json,pathlib,subprocess,tempfile,sys,numpy as np
root=pathlib.Path(__file__).resolve().parents[4];script=root/'src/task/ptv3-shared-wam-v3/tools/run/import_motion_overlay.py'


def test_motion_overlay_reuses_payloads_and_rejects_row_mismatch():
    with tempfile.TemporaryDirectory(dir=root/'tmp',prefix='motion-overlay-smoke-') as directory:
     base=pathlib.Path(directory);parent=base/'parent';old=parent/'native';old.mkdir(parents=True);(parent/'recipe').mkdir()
     source=base/'motion';source.mkdir();recipe=base/'new-recipe';recipe.mkdir();out=base/'overlay'
     def write(path,obj):path.write_text(json.dumps(obj))
     def read(path):return json.loads(path.read_text())
     rows=[]
     for i,split in enumerate(('train','dev')):
      payload=old/('payload-%d.npz'%i);payload.write_bytes(b'old payload remains unchanged')
      rows.append(dict(source_sequence='repeated-name',source_id=4,group_id='g%d'%i,objects=['cad'],target_slot=0,profile='profile',split=split,clock_verified=True,path=str(payload),payload_size_bytes=payload.stat().st_size,payload_mtime_ns=payload.stat().st_mtime_ns))
     geo=old/'cad.npz';geo.write_bytes(b'old CAD unchanged')
     oldmeta=dict(sequences=rows,objects=[dict(key='cad',path=str(geo))])
     write(old/'manifest.json',oldmeta);write(parent/'import-progress.json',dict(status='COMPLETED'));write(parent/'portable-reader-audit.json',dict(status='CPU_READER_PASS'))
     oldrecipe=dict(main3_root='/remote/existing-main3',main3_identity='same',source_counts={'0':10,'1':20,'2':30,'4':2})
     write(parent/'recipe/recipe.json',oldrecipe)
     np.savez(source/'windows.npz',verified_train=np.array([[0,3]]),verified_dev=np.array([[1,3]]),all_train=np.array([[0,3]]),all_dev=np.array([[1,3]]))
     np.savez(source/'excluded-parent-windows.npz',verified_train=np.array([[0,11]]))
     stamp=(source/'windows.npz').stat();meta=copy.deepcopy(oldmeta);meta.update(status='COMPLETED',training_allowed=False,motion_admission=dict(schema='simulator-impulse-prefix.v1'),window_index='windows.npz',window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns)
     write(source/'manifest.json',meta);write(source/'progress.json',dict(status='COMPLETED',input_manifest='/original/encoder/manifest.json'));write(source/'cleaning-summary.json',dict(parent='/original/encoder'))
     moments=dict(training_allowed=False,split='train',windows=1,input_manifest=str(source/'manifest.json'),source_moments={'4':dict(windows=1)})
     write(source/'source-moments.json',moments)
     newrecipe=copy.deepcopy(oldrecipe);newrecipe.update(native_root=str(source),native_index_stamp=[stamp.st_size,stamp.st_mtime_ns],native_engineering_recipe=dict(data_manifest=str(source/'manifest.json'),window_index_size_bytes=stamp.st_size,window_index_mtime_ns=stamp.st_mtime_ns,source_counts={'4':1}));newrecipe['source_counts']['4']=1
     write(recipe/'recipe.json',newrecipe);stats=dict(recipe=newrecipe,hand=dict(mean=[[1,2,3]],std=[[.1,.2,.3]]),scene=dict(mean=[[4,5,6]],std=[[.4,.5,.6]]));write(recipe/'horizon-stats.json',stats)
     original=(old/'manifest.json').read_bytes()
     command=[sys.executable,str(script),'--native-metadata',str(source),'--recipe-metadata',str(recipe),'--parent-import-root',str(parent),'--output',str(out)]
     result=subprocess.run(command,text=True,capture_output=True);assert result.returncode==0,result.stderr
     imported=read(out/'recipe/horizon-stats.json');bound=read(out/'recipe/recipe.json')
     assert imported['recipe']==bound and imported['hand']==stats['hand'] and imported['scene']==stats['scene']
     assert read(out/'native/manifest.json')['sequences'][0]['path']==rows[0]['path']
     assert (out/'native/windows.npz').stat().st_mtime_ns==stamp.st_mtime_ns
     assert (old/'manifest.json').read_bytes()==original
     assert not (out/'native/payloads').exists()
     # Positional identity guard must reject a misplaced row before writing output.
     broken=copy.deepcopy(meta);broken['sequences'][0]['source_sequence']='wrong';write(source/'manifest.json',broken)
     bad=subprocess.run(command[:-1]+[str(base/'rejected')],text=True,capture_output=True)
     assert bad.returncode!=0 and not (base/'rejected').exists()
     print('MOTION_OVERLAY_SMOKE_PASS: numeric/recipe/index bindings, zero payload copies, parent immutability, row identity rejection')
