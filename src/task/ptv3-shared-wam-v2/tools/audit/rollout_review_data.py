"""Read-only geometry/state loading and separate human review annotations."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import threading
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def xyzw_to_wxyz(quaternion):
    return np.asarray(quaternion)[..., [3, 0, 1, 2]]


def visual_meshes(urdf):
    """Bake visual origins/scales into link-local vertices, never joint poses."""
    urdf = Path(urdf)
    result = {}
    for link in ET.parse(urdf).getroot().findall('link'):
        parts = []
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            mesh_tag = geometry.find('mesh')
            if mesh_tag is not None:
                mesh = trimesh.load(urdf.parent / mesh_tag.attrib['filename'],
                                    force='mesh', process=False)
                mesh.apply_scale(np.fromstring(mesh_tag.get('scale', '1 1 1'), sep=' '))
            elif geometry.find('box') is not None:
                mesh = trimesh.creation.box(np.fromstring(geometry.find('box').get('size'), sep=' '))
            else:
                raise ValueError('Unsupported visual geometry: ' + str(urdf))
            origin = visual.find('origin')
            if origin is not None:
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                mesh.apply_transform(transform)
            parts.append(mesh)
        if parts:
            mesh = trimesh.util.concatenate(parts)
            # STL duplicates triangle vertices. Deduplicate without decimation.
            mesh.merge_vertices()
            result[link.attrib['name']] = mesh
    return result


class Corpus:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if (self.manifest.get('status') != 'COMPLETED'
                or not self.manifest.get('capture_current_observation')):
            raise ValueError('Completed native-current rollout corpus required')
        self.records = self.manifest['records']
        self.contract = self.manifest['observation_contract']
        self.fps = self.manifest['fps']
        self.by_name = {r['sequence']: r for r in self.records}
        batch = self.root / self.records[0]['path']
        batch_manifest = json.loads((batch.parents[1] / 'manifest.json').read_text())
        self.geometric_root = Path(batch_manifest['geometric_plan_root'])

    def load(self, name):
        record = self.by_name[name]
        path = (self.root / record['path'] / 'trajectory.npz').resolve()
        if self.root not in path.parents:
            raise ValueError('Trajectory escapes corpus')
        keys = ('body_state', 'body_contact_forces', 'obj', 'reference_obj', 'reference_current')
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in keys}
        ref = data['reference_current']
        data['reference_hand'] = ref[:, 119:167].reshape(-1, 16, 3)
        data['object_error'] = np.linalg.norm(data['obj'][:, :3, 3] - data['reference_obj'][:, :3, 3], axis=-1)
        data['hand_error'] = np.linalg.norm(
            data['body_state'][:, self.contract['key_body_ids'], :3] - data['reference_hand'], axis=-1).mean(-1)
        data['contact'] = (np.abs(data['body_contact_forces'][:, self.contract['contact_body_ids']]) > .1).any(-1)
        data['reference_contact'] = ref[:, 168:184][:, [3, 6, 9, 12, 15]]
        return record, data

    def table_pose(self, name):
        # Native _load_table uses these original geometric fields at frame zero.
        # Pure file reading on CPU; no model/simulation is run here.
        import torch
        source = self.geometric_root / name / 'interaction_hand_inspire.pt'
        raw = torch.load(source, map_location='cpu', weights_only=False)
        return raw[0, 238:245].numpy().copy()


class ReviewJournal:
    """One immutable JSON file per annotation; source data is never rewritten."""
    def __init__(self, output, corpus):
        self.output = Path(output).resolve()
        self.corpus = corpus
        if self.output == corpus.root or corpus.root in self.output.parents:
            raise ValueError('Review output must be separate from source corpus')
        self.output.mkdir(parents=True, exist_ok=True)
        self.annotations = self.output / 'annotations'
        self.annotations.mkdir(exist_ok=True)
        self.lock = threading.Lock()

    def save(self, sequence, start, end_inclusive, verdict, reason, note):
        record = self.corpus.by_name[sequence]
        if int(start) != start or int(end_inclusive) != end_inclusive:
            raise ValueError('Frame boundaries must be integers')
        start, end_inclusive = int(start), int(end_inclusive)
        if not 0 <= start <= end_inclusive < record['frames']:
            raise ValueError('Invalid interval: require 0 <= start <= end < frame count')
        if verdict not in ('usable', 'unusable', 'uncertain') or len(note) > 2000:
            raise ValueError('Invalid verdict or note exceeds 2000 characters')
        row = dict(schema='ref2dex.rollout-human-review.v1', annotation_id=uuid.uuid4().hex,
                   created_at=datetime.now(timezone.utc).isoformat(), source_root=str(self.corpus.root),
                   source_run_id=self.corpus.root.name, sequence=sequence, split=record['split'],
                   start_frame=start, end_frame_exclusive=end_inclusive + 1,
                   interval_convention='[start_frame,end_frame_exclusive)', fps=self.corpus.fps,
                   verdict=verdict, reason=reason, note=note)
        with self.lock:
            if sum(1 for _ in self.annotations.glob('*.json')) >= 10000:
                raise ValueError('Review annotation budget reached (10000)')
            path = self.annotations / (row['annotation_id'] + '.json')
            with path.open('x') as handle:
                json.dump(row, handle, indent=2, ensure_ascii=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
        return path

    def count(self, sequence):
        return sum(json.loads(p.read_text())['sequence'] == sequence
                   for p in self.annotations.glob('*.json'))
