# Machine and runtime memory

## Remote 4×4090 node

Last verified: 2026-10-17.

- SSH endpoint: `wbcd@127.0.0.1`, port `22023`, key `/home2/wyy/.ssh/id_ed25519`.
  The known-hosts file used by this repository is
  `tmp/ssh-4-4090/known_hosts`.  OpenSSH uses `-p 22023`; `scp` uses the
  uppercase `-P 22023`.
- The node is `deepcybo-4-4090-48` with four 49,140 MiB GPUs.  GPU 0 is the
  default for short probes when `nvidia-smi` reports it idle.
- The remote repository used for current WAM work is
  `/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v3-nine-source-20261015-r1/training-code`.
  Do not assume `/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent` exists on that node.
- Python: `/home/wbcd/miniconda3/envs/graspenv/bin/python` (Torch 2.0.1/cu118).
  The environment itself does **not** import `isaacgym`; Isaac Gym is available
  at `/home/wbcd/workspace/dex/third_party/isaacgym/python` and must be added to
  `PYTHONPATH`/`sys.path` before importing it.
- Remote DExplore checkout: `/home/wbcd/workspace/dex/retarget/dexplore`.
  Its official Inspire checkpoint is
  `/home/wbcd/workspace/dex/retarget/dexplore/checkpoint/inspire.pth`, SHA256
  `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`.
  The shared training tree also has `third_party/DExplore`; use the checkpoint
  whose hash is explicitly checked by the runner.
- A remote motion available for a fresh DExplore check is
  `/home/wbcd/workspace/dex/retarget/dexplore/data/s1_airplane_fly_1/interaction_hand_inspire.pt`.

## Local GPU safety

- On 2026-10-17, local GPUs 0–7 were occupied by `/home2/monica/.../morl-baselines/bin/python`
  processes, each using about 0.4 GiB.  Do not kill or reuse those processes;
  use the remote node or wait for an explicitly idle local GPU.

## Experiment transfer notes

- Remote output and cache paths must remain under the remote training tree or
  its NAS output root; do not write experiment artifacts to `/tmp`.
- Large checkpoint transfers through the forwarded SSH endpoint can be
  truncated when sent as one long command.  Verify size and SHA256 after every
  transfer; if needed, split into small repository-owned chunks and reassemble
  remotely.
- A remote copy of the repository may be on an older detached commit.  Sync the
  exact task-local runner and record its local Git commit before running a
  probe; never infer version identity from the remote checkout alone.
