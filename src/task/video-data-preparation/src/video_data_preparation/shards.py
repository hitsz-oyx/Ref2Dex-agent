"""Streaming HDF5 shards and immutable, session-safe dataset snapshots."""
from collections import OrderedDict
import json
import os
from pathlib import Path
import uuid
import numpy as np

from . import SCHEMA
from .io import atomic_json, confined_path
from .sample_contract import FIXED_FIELDS, validate_sample


class ShardWriter:
    def __init__(self, path, split, session, profile_id, max_bytes=256*2**20):
        import h5py
        self.path = Path(path)
        if split not in ("train", "val") or max_bytes < 1 or not session:
            raise ValueError("valid split/session and positive shard cap required")
        if self.path.exists() or self.path.with_suffix(".json").exists():
            raise FileExistsError("immutable shard already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.part = self.path.with_name(self.path.name+"."+uuid.uuid4().hex+".partial")
        self.file = h5py.File(self.part, "x")
        self.file.attrs.update(schema=SCHEMA, split=split, session=session, profile_id=profile_id)
        self.split, self.session, self.profile_id = split, session, profile_id
        self.count, self.bytes, self.max_bytes, self.ticks = 0, 0, max_bytes, []
        self.episode_keys = set()
        self.admitted = True
        for key, (_, kind) in FIXED_FIELDS.items():
            shape = FIXED_FIELDS[key][0]
            dtype = {"f": np.float32, "b": bool, "i": np.int64}[kind]
            self.file.create_dataset(key, shape=(0, *shape), maxshape=(None, *shape),
                                     dtype=dtype, chunks=(1, *shape), compression="lzf")
        for key, tail, dtype in (("xyz", (3,), np.float32), ("features", (18,), np.float32),
                                  ("scene_object", (), np.int64)):
            self.file.create_dataset(key, shape=(0, *tail), maxshape=(None, *tail), dtype=dtype,
                                     chunks=(2048, *tail), compression="lzf")
        self.file.create_dataset("offsets", data=np.array([0], np.int64), maxshape=(None,), chunks=True)
        self.file.create_dataset("metadata", shape=(0,), maxshape=(None,), chunks=True,
                                 dtype=h5py.string_dtype("utf-8"))
        self.file.create_dataset("track_uv", shape=(0, 25, 32, 2), maxshape=(None, 25, 32, 2),
                                 chunks=(1, 25, 32, 2), dtype=np.float32, compression="lzf")
        self.file.create_dataset("track_valid", shape=(0, 25, 32), maxshape=(None, 25, 32),
                                 chunks=(1, 25, 32), dtype=bool, compression="lzf")

    def append(self, sample, metadata, audit):
        validate_sample(sample)
        if metadata.get("session") != self.session or not metadata.get("causal_observation"):
            raise ValueError("shard session differs or observation is noncausal")
        if int(sample["sample_id"][1]) != metadata["tick"] or metadata["tick"] in self.ticks:
            raise ValueError("duplicate/mismatched native sample clock")
        for key in FIXED_FIELDS:
            dataset = self.file[key]
            dataset.resize(self.count+1, axis=0)
            dataset[self.count] = sample[key]
        old = int(self.file["offsets"][-1])
        size = len(sample["xyz"])
        for key in ("xyz", "features", "scene_object"):
            dataset = self.file[key]
            dataset.resize(old+size, axis=0)
            dataset[old:old+size] = sample[key]
        offsets = self.file["offsets"]
        offsets.resize(self.count+2, axis=0)
        offsets[-1] = old+size
        for key, value in (("metadata", json.dumps(metadata, ensure_ascii=False, allow_nan=False)),
                            ("track_uv", audit["track_uv"]), ("track_valid", audit["track_valid"])):
            dataset = self.file[key]
            dataset.resize(self.count+1, axis=0)
            dataset[self.count] = value
        self.admitted &= bool(metadata.get("training_allowed"))
        self.ticks.append(metadata["tick"])
        self.episode_keys.add(metadata["episode_key"])
        self.count += 1
        self.bytes += sum(np.asarray(sample[key]).nbytes for key in (*FIXED_FIELDS, "xyz", "features", "scene_object"))
        self.bytes += audit["track_uv"].nbytes + audit["track_valid"].nbytes + len(json.dumps(metadata).encode())

    @property
    def full(self):
        return self.bytes >= self.max_bytes

    def publish(self):
        if not self.count:
            raise ValueError("cannot publish empty shard")
        self.file.flush()
        self.file.close()
        with self.part.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(self.part, self.path)
        except FileExistsError:
            raise FileExistsError("another worker already published this shard")
        finally:
            self.part.unlink(missing_ok=True)
        receipt = dict(schema=SCHEMA, status="PUBLISHED", path=self.path.name,
                       bytes=self.path.stat().st_size, samples=self.count, split=self.split,
                       session=self.session, ticks=self.ticks, episode_keys=sorted(self.episode_keys), profile_id=self.profile_id,
                       profile="geometry-only", causal_observation=True, training_allowed=self.admitted,
                       fps=30, history=4, horizon=24, object_points=32)
        atomic_json(self.path.with_suffix(".json"), receipt, exclusive=True)
        return receipt

    def abort(self):
        if self.file.id.valid:
            self.file.close()
        # Retain only this worker's incomplete artifact for inspection/recovery.


def freeze_snapshot(root, output, require_both=True, engineering=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.parent != root:
        raise ValueError("snapshot must be directly inside corpus root for relocatable paths")
    shards, sessions, profiles, samples = [], {"train": set(), "val": set()}, set(), set()
    for path in sorted((root/"shards").glob("*.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != SCHEMA or receipt.get("status") != "PUBLISHED":
            continue
        if not receipt.get("training_allowed") or not receipt.get("causal_observation"):
            raise ValueError("snapshot contains an unadmitted/noncausal shard")
        split = receipt["split"]
        if split not in sessions or not receipt["samples"]:
            raise ValueError("invalid shard split or sample count")
        shard_path = confined_path(root/"shards", receipt["path"])
        if not shard_path.is_file() or shard_path.stat().st_size != receipt["bytes"]:
            raise ValueError("published shard missing/size changed")
        if shard_path.with_suffix(".json") != path:
            raise ValueError("receipt filename differs from shard")
        if len(receipt.get("episode_keys", [])) != 1:
            raise ValueError("one episode per shard required")
        ids = {(receipt["episode_keys"][0], tick) for tick in receipt["ticks"]}
        if len(ids) != receipt["samples"] or ids & samples:
            raise ValueError("duplicate native windows in published shards")
        samples.update(ids)
        sessions[split].add(receipt["session"])
        profiles.add(receipt["profile_id"])
        shards.append(dict(receipt, path=str(shard_path.relative_to(root))))
    if not shards or len(profiles) != 1:
        raise ValueError("one nonempty, fixed preprocessing profile required")
    if sessions["train"] & sessions["val"]:
        raise ValueError("recording session leaked across train/dev")
    if require_both and (not sessions["train"] or not sessions["val"]):
        raise ValueError("training snapshot needs independent train and dev sessions")
    snapshot = dict(schema=SCHEMA, status="FROZEN", training_allowed=True, causal_observation=True,
                    engineering_only=engineering,
                    profile_id=next(iter(profiles)), profile="geometry-only", fps=30, history=4,
                    horizon=24, object_points=32, source="egodex", source_id=3,
                    shards=shards, counts={s: sum(x["samples"] for x in shards if x["split"] == s) for s in sessions})
    atomic_json(output, snapshot, exclusive=True)
    return snapshot


def recover_publications(root, profile_id):
    """Recover a crash after data rename but before receipt publication.

    Only final .h5 files in this corpus are adopted, after validating every row;
    interrupted .partial files remain excluded and retained as failed evidence.
    """
    import h5py
    recovered = 0
    for path in sorted((Path(root)/"shards").glob("*.h5")):
        if path.with_suffix(".json").exists():
            continue
        with h5py.File(path, "r") as f:
            if f.attrs.get("schema") != SCHEMA or f.attrs.get("profile_id") != profile_id:
                raise ValueError("orphan shard belongs to another schema/profile")
            n = len(f["sample_id"])
            metadata = []
            for row in range(n):
                lo, hi = map(int, f["offsets"][row:row+2])
                sample = {k: np.asarray(f[k][row]) for k in FIXED_FIELDS}
                sample.update({k: f[k][lo:hi] for k in ("xyz", "features", "scene_object")})
                validate_sample(sample)
                meta = json.loads(f["metadata"][row])
                if not meta.get("causal_observation") or (not meta.get("training_allowed") and
                        meta.get("target_verification") != "automatic-provisional"):
                    raise ValueError("cannot adopt unqualified orphan")
                metadata.append(meta)
            if (not metadata or len({m["episode_key"] for m in metadata}) != 1
                    or any(m["session"] != f.attrs["session"] for m in metadata)):
                raise ValueError("orphan has no unique episode")
            receipt = dict(schema=SCHEMA, status="PUBLISHED", path=path.name, bytes=path.stat().st_size,
                samples=n, split=f.attrs["split"], session=f.attrs["session"], profile_id=profile_id,
                ticks=[m["tick"] for m in metadata], episode_keys=[metadata[0]["episode_key"]],
                profile="geometry-only", causal_observation=True,
                training_allowed=all(m.get("training_allowed", False) for m in metadata),
                fps=30, history=4, horizon=24, object_points=32)
        atomic_json(path.with_suffix(".json"), receipt, exclusive=True)
        recovered += 1
    return recovered


class ShardDataset:
    """Small frozen index; HDF5 handles open in the reading process only."""
    def __init__(self, snapshot, split, cache_shards=4, engineering=False):
        if split not in ("train", "val") or not 1 <= cache_shards <= 32:
            raise ValueError("train/val split and bounded shard cache required")
        self.snapshot = Path(snapshot).resolve()
        self.root = self.snapshot.parent
        self.meta = json.loads(self.snapshot.read_text())
        if (self.meta.get("schema") != SCHEMA or self.meta.get("status") != "FROZEN"
                or not self.meta.get("training_allowed") or not self.meta.get("causal_observation")
                or tuple(self.meta.get(k) for k in ("fps", "history", "horizon", "object_points")) != (30, 4, 24, 32)):
            raise ValueError("qualified immutable EgoDex snapshot required")
        if self.meta.get("engineering_only") and not engineering:
            raise ValueError("engineering corpus requires an explicit engineering reader")
        all_shards = self.meta["shards"]
        train_sessions = {s["session"] for s in all_shards if s["split"] == "train"}
        val_sessions = {s["session"] for s in all_shards if s["split"] == "val"}
        if train_sessions & val_sessions:
            raise ValueError("snapshot session leakage")
        self.shards = [s for s in all_shards if s["split"] == split]
        self.ends = np.cumsum([s["samples"] for s in self.shards])
        if not len(self.ends) or self.ends[-1] < 1:
            raise ValueError("empty snapshot split")
        for shard in self.shards:
            path = confined_path(self.root, shard["path"])
            if path.stat().st_size != shard["bytes"] or shard["profile_id"] != self.meta["profile_id"]:
                raise ValueError("shard size/profile differs from snapshot")
        self.cache_shards, self._handles, self._pid = cache_shards, OrderedDict(), os.getpid()
        self.groups = {}
        for i, shard in enumerate(self.shards):
            start = int(self.ends[i-1]) if i else 0
            key = (shard["session"], shard["episode_keys"][0])
            self.groups.setdefault(key, []).append((start, int(self.ends[i])))
        self.sampling_sessions = sorted({key[0] for key in self.groups})
        self.sampling_groups = []
        for session in self.sampling_sessions:
            entries = []
            for key in sorted(k for k in self.groups if k[0] == session):
                spans = np.asarray(self.groups[key], np.int64)
                entries.append((spans, np.cumsum(spans[:, 1]-spans[:, 0])))
            self.sampling_groups.append(entries)

    def __len__(self):
        return int(self.ends[-1])

    def _locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = int(np.searchsorted(self.ends, index, side="right"))
        return shard, int(index-(self.ends[shard-1] if shard else 0))

    def _file(self, shard):
        import h5py
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if shard in self._handles:
            self._handles.move_to_end(shard)
            return self._handles[shard]
        info = self.shards[shard]
        f = h5py.File(confined_path(self.root, info["path"]), "r")
        if (f.attrs.get("schema") != SCHEMA or f.attrs.get("split") != info["split"]
                or f.attrs.get("session") != info["session"] or f.attrs.get("profile_id") != info["profile_id"]
                or len(f["sample_id"]) != info["samples"]):
            f.close()
            raise ValueError("shard header differs from frozen snapshot")
        self._handles[shard] = f
        while len(self._handles) > self.cache_shards:
            self._handles.popitem(last=False)[1].close()
        return f

    def __getitem__(self, index):
        shard, row = self._locate(index)
        f = self._file(shard)
        lo, hi = map(int, f["offsets"][row:row+2])
        sample = {key: np.asarray(f[key][row]) for key in FIXED_FIELDS}
        sample.update({key: f[key][lo:hi] for key in ("xyz", "features", "scene_object")})
        return validate_sample(sample)

    def sample_indices(self, count, rng):
        """Uniform sessions, then episodes; long videos do not dominate."""
        result = []
        for _ in range(count):
            episodes = self.sampling_groups[int(rng.integers(len(self.sampling_groups)))]
            spans, ends = episodes[int(rng.integers(len(episodes)))]
            offset = int(rng.integers(ends[-1]))
            selected = int(np.searchsorted(ends, offset, side="right"))
            preceding = int(ends[selected-1]) if selected else 0
            result.append(int(spans[selected, 0]+offset-preceding))
        return np.asarray(result, np.int64)

    def audit(self, index):
        shard, row = self._locate(index)
        f = self._file(shard)
        raw = f["metadata"][row]
        return dict(metadata=json.loads(raw), track_uv=f["track_uv"][row], track_valid=f["track_valid"][row])

    def close(self):
        for f in self._handles.values():
            f.close()
        self._handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = OrderedDict()
        return state
