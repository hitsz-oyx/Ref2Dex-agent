"""Single-host SQLite jobs with bounded leases and split-safe READY ingestion."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid


class JobLedger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")  # Compatible with NAS lacking WAL locks.
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (key TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                        "state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, token TEXT, "
                        "lease_until REAL, receipt TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS offsets (path TEXT PRIMARY KEY, offset INTEGER NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS windows (profile TEXT, episode TEXT, tick INTEGER, shard TEXT, "
                        "PRIMARY KEY(profile,episode,tick))")

    def ingest(self, ready_path, profile_id):
        """Commit full JSONL lines and cursor in the same transaction."""
        path = str(Path(ready_path).resolve())
        if not Path(path).exists():
            return 0  # Producer may not have published its first complete pair yet.
        self.db.execute("BEGIN IMMEDIATE")
        count = 0
        try:
            row = self.db.execute("SELECT offset FROM offsets WHERE path=?", (path,)).fetchone()
            offset = row[0] if row else 0
            if Path(path).stat().st_size < offset:
                raise ValueError("READY producer file was truncated")
            with open(path, "rb") as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    payload = json.loads(line)
                    if payload.get("release_split") == "official_train":
                        if not payload.get("key") or not payload.get("raw_hdf5"):
                            raise ValueError("READY record missing raw identity")
                        key = profile_id + ":" + payload["key"]
                        count += self.db.execute("INSERT OR IGNORE INTO jobs (key,payload,state) VALUES (?,?,?)",
                            (key, json.dumps(payload), "READY")).rowcount
                    offset = stream.tell()
            self.db.execute("INSERT OR REPLACE INTO offsets VALUES (?,?)", (path, offset))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        return count

    def claim(self, lease_seconds=1800, max_attempts=2, now=None):
        if lease_seconds <= 0 or max_attempts < 1:
            raise ValueError("positive bounded lease/retries required")
        now = time.time() if now is None else now
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute("UPDATE jobs SET state='RETRYABLE',token=NULL WHERE "
                            "state='PROCESSING' AND lease_until<=?", (now,))
            self.db.execute("UPDATE jobs SET state='DEFERRED' WHERE state='RETRYABLE' AND attempts>=?", (max_attempts,))
            job = self.db.execute("SELECT * FROM jobs WHERE state IN ('READY','RETRYABLE') AND attempts<? "
                                  "ORDER BY rowid LIMIT 1", (max_attempts,)).fetchone()
            if job is None:
                self.db.execute("COMMIT")
                return None
            token = uuid.uuid4().hex
            self.db.execute("UPDATE jobs SET state='PROCESSING',attempts=attempts+1,token=?,lease_until=? WHERE key=?",
                            (token, now+lease_seconds, job["key"]))
            self.db.execute("COMMIT")
            return dict(key=job["key"], payload=json.loads(job["payload"]), token=token)
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def heartbeat(self, job, lease_seconds=1800):
        changed = self.db.execute("UPDATE jobs SET lease_until=? WHERE key=? AND token=? AND state='PROCESSING'",
                                  (time.time()+lease_seconds, job["key"], job["token"])).rowcount
        if changed != 1:
            raise RuntimeError("lost job lease")

    def finish(self, job, state, receipt):
        if state not in ("PUBLISHED", "RETRYABLE", "REJECTED", "DEFERRED"):
            raise ValueError("invalid terminal/retry state")
        changed = self.db.execute("UPDATE jobs SET state=?,receipt=?,token=NULL,lease_until=NULL "
            "WHERE key=? AND token=? AND state='PROCESSING'",
            (state, json.dumps(receipt), job["key"], job["token"])).rowcount
        if changed != 1:
            raise RuntimeError("stale worker cannot publish job")

    def counts(self):
        return dict(self.db.execute("SELECT state,count(*) FROM jobs GROUP BY state"))

    def register_shard(self, receipt):
        if receipt.get("status") != "PUBLISHED" or len(receipt.get("episode_keys", [])) != 1:
            raise ValueError("published single-episode shard required")
        episode = receipt["episode_keys"][0]
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for tick in receipt["ticks"]:
                old = self.db.execute("SELECT shard FROM windows WHERE profile=? AND episode=? AND tick=?",
                    (receipt["profile_id"], episode, tick)).fetchone()
                if old is not None and old[0] != receipt["path"]:
                    raise ValueError("native window already published in another shard")
                self.db.execute("INSERT OR IGNORE INTO windows VALUES (?,?,?,?)",
                    (receipt["profile_id"], episode, tick, receipt["path"]))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def published_ticks(self, profile, episode):
        return {row[0] for row in self.db.execute("SELECT tick FROM windows WHERE profile=? AND episode=?", (profile, episode))}

    def close(self):
        self.db.close()


def stable_split(session, registry=None, seed=283):
    canonical = session[:-4] if session.endswith(".mov") else session
    if registry is not None and (session in registry or canonical in registry):
        if session in registry and canonical in registry and registry[session] != registry[canonical]:
            raise ValueError("conflicting canonical/literal session split")
        split = registry[session] if session in registry else registry[canonical]
        if split not in ("train", "val"):
            raise ValueError("split registry must contain train/val")
        return split
    # Same grouping recipe as the existing full reconstruction consumer.
    session = canonical
    return "val" if int(hashlib.sha256((str(seed)+":"+session).encode()).hexdigest()[:8], 16) % 5 == 0 else "train"
