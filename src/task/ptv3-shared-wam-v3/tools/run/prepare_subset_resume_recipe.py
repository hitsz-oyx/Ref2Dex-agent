"""Bind a reviewed data subset to the parent's unchanged numerical normalizer."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','subset','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve()
    if out.exists():raise ValueError('fresh continuation recipe required')
    old=json.loads((a.reference/'recipe.json').read_text());new=json.loads((a.subset/'recipe.json').read_text())
    stats=json.loads((a.reference/'horizon-stats.json').read_text())
    if stats['recipe']!=old or old['main3_identity']!=new['main3_identity']:raise ValueError('reference mismatch')
    if not new['admission']['motion_admission'].get('manual_decisions'):raise ValueError('reviewed subset required')
    new['source_probabilities']=copy.deepcopy(old['source_probabilities'])
    new['sampling']['source']='fixed parent source probabilities; uniform new native frozen windows'
    reference=dict(recipe=str(a.reference.resolve()),native_root=old['native_root'],
        policy='Frozen original TRAIN normalizer for exact model/optimizer continuation; fresh-fit statistics separately retained.',
        fresh_statistics_recipe=str(a.subset.resolve()))
    stats.update(recipe=new,normalization_reference=reference)
    out.mkdir(parents=True)
    for name,value in [('recipe.json',new),('horizon-stats.json',stats)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    for name in ('source-moments.json','main3-train-stat-indices.npz'):shutil.copy2(a.subset/name,out/name)
    (out/'manifest.json').write_text(json.dumps(dict(status='COMPLETED',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        reference_recipe=str(a.reference.resolve()),fresh_statistics_recipe=str(a.subset.resolve()),
        parent_recipe=str(a.reference.resolve()),native_root=new['native_root'],normalization_reference=reference),indent=2))
    print(json.dumps(dict(status='COMPLETED',counts=new['source_counts'],normalizer_unchanged=True)))


if __name__=='__main__':main()
