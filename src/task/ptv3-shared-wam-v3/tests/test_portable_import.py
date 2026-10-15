"""A relocation changes locations only, retaining exact stamps and recipe math."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TASK = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[4]


class PortableImportTest(unittest.TestCase):
    def test_roundtrip_preserves_bindings_flags_arrays_and_nanoseconds(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp',prefix='portable-test-') as directory:
            base=Path(directory);native=base/'original-native';native.mkdir()
            main3=base/'existing-main3';(main3/'processed').mkdir(parents=True)
            recipe_dir=base/'recipe';recipe_dir.mkdir()
            def write(path,value):path.write_text(json.dumps(value))
            def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
            payload=native/'a.npz';payload.write_bytes(b'opaque payload unchanged')
            index=native/'windows.npz';index.write_bytes(b'frozen index bytes')
            source=native/'source.npz';source.write_bytes(b'source hash')
            meta=dict(window_index='windows.npz',window_index_size_bytes=index.stat().st_size,
                window_index_mtime_ns=index.stat().st_mtime_ns,training_allowed=False,
                sequences=[dict(path=str(payload),payload_size_bytes=payload.stat().st_size,
                    payload_mtime_ns=payload.stat().st_mtime_ns)],objects=[dict(path=str(payload))])
            write(native/'manifest.json',meta)
            write(native/'source-moments.json',dict(input_manifest=str(native/'manifest.json'),training_allowed=False))
            for name in ('progress.json','cleaning-summary.json'):write(native/name,dict(status='COMPLETED'))
            (native/'excluded-parent-windows.npz').write_bytes(b'exclusions')
            # Reused main3 remains byte-identical and is checked by existing source/index identities.
            write(main3/'processed/source.json',{})
            source_root=base/'source';(source_root/'processed').mkdir(parents=True)
            (source_root/'processed/manifest.json').write_bytes(b'{}')
            main_meta=dict(normalization_source_manifest_sha256='frozen-main3',sources=[dict(
                root=str(source_root),manifest_sha256=sha(source_root/'processed/manifest.json'),
                indices={s:dict(path=str(source),sha256=sha(source)) for s in ('train','val','test')})])
            write(main3/'processed/manifest.json',main_meta)
            recipe=dict(native_root=str(native),main3_root=str(main3),training_allowed=True,
                native_index_stamp=[index.stat().st_size,index.stat().st_mtime_ns],
                main3_identity='frozen-main3',source_counts={'4':1},
                native_engineering_recipe=dict(data_manifest=str(native/'manifest.json')))
            stats=dict(recipe=recipe,hand=dict(mean=[[1,2,3]],std=[[.1,.2,.3]]),scene=dict(mean=[[4,5,6]],std=[[.4,.5,.6]]))
            write(recipe_dir/'recipe.json',recipe);write(recipe_dir/'horizon-stats.json',stats)
            original_native=(native/'manifest.json').read_bytes();original_main=(main3/'processed/manifest.json').read_bytes()
            export=base/'export';target=base/'import'
            for script,args in [('export_portable_nine.py',['--native',str(native),'--recipe',str(recipe_dir),'--output',str(export)]),
                    ('import_portable_nine.py',['--export',str(export),'--main3',str(main3),'--output',str(target)])]:
                subprocess.run([sys.executable,str(TASK/'tools/run'/script)]+args,check=True,capture_output=True)
            imported=json.loads((target/'recipe/horizon-stats.json').read_text())
            bound=json.loads((target/'recipe/recipe.json').read_text())
            self.assertEqual(imported['recipe'],bound)
            self.assertEqual(imported['hand'],stats['hand']);self.assertEqual(imported['scene'],stats['scene'])
            self.assertEqual(bound['native_engineering_recipe']['data_manifest'],str(target/'native/manifest.json'))
            imported_meta=json.loads((target/'native/manifest.json').read_text())
            moved=target/'native'/imported_meta['sequences'][0]['path']
            self.assertEqual(moved.read_bytes(),payload.read_bytes())
            self.assertEqual(moved.stat().st_mtime_ns,payload.stat().st_mtime_ns)
            self.assertFalse(imported_meta['training_allowed']);self.assertTrue(bound['training_allowed'])
            self.assertEqual((native/'manifest.json').read_bytes(),original_native)
            self.assertEqual((main3/'processed/manifest.json').read_bytes(),original_main)


if __name__=='__main__':unittest.main()
