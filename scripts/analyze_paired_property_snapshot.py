"""Within-group SDK property equality; never infer a unique physics cause."""
import argparse
import json
from pathlib import Path
import numpy as np


def flatten(value):
    if isinstance(value, dict):
        return sum((flatten(value[k]) for k in sorted(value)), [])
    if isinstance(value, list):
        return sum((flatten(v) for v in value), [])
    return [float(value)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    root = a.directory.resolve()
    records = json.loads((root / 's605/sdk_properties.json').read_text())
    assert len(records) == 768
    groups = {}
    for row in records:
        groups.setdefault(row['cluster'], []).append(row)
    maxima = {key:0. for key in ('dof','hand_bodies','object_bodies','hand_shapes','object_shapes')}
    for group in groups.values():
        assert len(group) == 4
        for key in maxima:
            arrays = [flatten(x[key]) for x in group]
            assert all(len(x) == len(arrays[0]) for x in arrays)
            maxima[key] = max(maxima[key], float(np.abs(np.array(arrays) - arrays[0]).max()))
    result = dict(run_status='COMPLETED',engineering_only=True,all768_actor_properties_read=True,
                  within_group_sdk_property_maximum=maxima,physics_steps=0,model_forward_calls=0,
                  per_actor_property_difference_found=any(v != 0 for v in maxima.values()),
                  environment_origin_max_abs=float(np.abs([x['origin'] for x in records]).max()),
                  unique_environment_origins=len(set(tuple(x['origin']) for x in records)),
                  unique_causal_explanation_not_established=True)
    assert not (root / 'results.json').exists()
    (root / 'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
