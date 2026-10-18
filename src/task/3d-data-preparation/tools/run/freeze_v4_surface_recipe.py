"""Bind existing dense NPZ contents to a new recipe without resampling data."""
import argparse
import json
from pathlib import Path
import sys

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from pose_mesh_data.dense_surface import SCHEMA, file_sha256, read_surface_manifest


def freeze(base_recipe, output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    recipe=json.loads((Path(base_recipe)/'recipe.json').read_text())
    stats=json.loads((Path(base_recipe)/'horizon-stats.json').read_text())
    source=Path(recipe['surface_geometry_manifest'])
    # A v1 upgrade writes a fresh manifest. A bound v2 input must still match
    # its original digest; it cannot silently be rebound after modification.
    previous=json.loads(source.read_text())
    if previous['schema']==SCHEMA:
        previous=read_surface_manifest(source,recipe['surface_geometry_sha256'])
    elif previous['schema']!='ref2dex.wam-v4-dense-surface.v1' or previous['status']!='COMPLETED':
        raise ValueError('known completed dense geometry required')
    rows=[]
    for row in previous['objects']:
        path=(source.parent/row['path']).resolve()
        digest=file_sha256(path)
        if previous['schema']==SCHEMA and digest!=row.get('sha256'):
            raise ValueError('existing bound dense surface payload SHA-256 mismatch')
        rows.append(dict(row,path=str(path),sha256=digest))
    manifest=dict(previous,schema=SCHEMA,objects=rows,parent_manifest=str(source),
                  identity='SHA-256 of each NPZ, bound by recipe manifest SHA-256')
    dest=output/'surface-manifest.json'
    dest.write_text(json.dumps(manifest,indent=2)+'\n')
    recipe['surface_geometry_manifest']=str(dest)
    recipe['surface_geometry_sha256']=file_sha256(dest)
    recipe['data_profile']='nine-source-v4-target-object-content-bound.v1'
    stats['recipe']=recipe
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    (output/'horizon-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    return dict(status='COMPLETED',objects=len(rows),geometry_resampled=False,
                surface_geometry_sha256=recipe['surface_geometry_sha256'],output=str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-recipe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.base_recipe,a.output)),flush=True)
