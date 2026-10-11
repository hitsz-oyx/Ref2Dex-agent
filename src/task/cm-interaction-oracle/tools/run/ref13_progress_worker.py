"""Run the historical collector in its Torch2.4 runtime with a pinned actor.

Only the uniform torch.compile key prefix is translated; tensor values and
the external native observation normalizer are unchanged.
"""
import importlib.util
from pathlib import Path


def main():
    path = Path(__file__).with_name('collect_oracle_y_candidates.py')
    spec = importlib.util.spec_from_file_location('ref13_native_collector', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)  # Isaac Gym must precede Torch.
    if collector.torch.__version__ != '2.4.1+cu121':
        raise RuntimeError('ref13 worker requires preserved Torch2.4.1+cu121 runtime')
    original_load = collector.original.torch_ext.load_checkpoint

    def load(filename):
        checkpoint = original_load(filename)
        state = checkpoint['model']
        compiled = [key.startswith('_orig_mod.') for key in state]
        if any(compiled) and not all(compiled):
            raise ValueError('mixed torch.compile checkpoint keys')
        if all(compiled):
            checkpoint = dict(checkpoint)
            checkpoint['model'] = {key[len('_orig_mod.'):]: value for key, value in state.items()}
        return checkpoint

    collector.original.torch_ext.load_checkpoint = load
    collector.main()


if __name__ == '__main__':
    main()
