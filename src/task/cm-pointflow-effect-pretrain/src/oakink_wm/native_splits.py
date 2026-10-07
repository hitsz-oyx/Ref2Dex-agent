"""Author-supplied GRAB object hold-outs and ARCTIC protocol membership."""
import json
from pathlib import Path

GRAB_TEST = frozenset(('mug', 'wineglass', 'camera', 'binoculars', 'fryingpan', 'toothpaste'))
GRAB_VAL = frozenset(('apple', 'toothbrush', 'elephant', 'hand'))


def grab_split(sequence):
    obj = Path(sequence).name.split('_')[0]
    return 'test' if obj in GRAB_TEST else 'val' if obj in GRAB_VAL else 'train'


def arctic_protocol(path):
    data = json.loads(Path(path).read_text())
    if set(data) != {'train', 'val', 'test'}:
        raise ValueError('ARCTIC protocol must declare train/val/test')
    membership = {}
    for split, sequences in data.items():
        for seq in sequences:
            name = str(seq).replace('.mano.npy', '').replace('.object.npy', '')
            if name in membership:
                raise ValueError('duplicate ARCTIC sequence across protocol splits: ' + name)
            membership[name] = split
    return membership


def official_split(source, sequence, protocol):
    if source == 'grab':
        return grab_split(sequence)
    if source == 'arctic':
        name = str(sequence).replace('.mano.npy', '').replace('.object.npy', '')
        if name not in protocol:
            raise ValueError('ARCTIC sequence missing from official protocol: ' + name)
        return protocol[name]
    raise ValueError('unknown native source: ' + str(source))
