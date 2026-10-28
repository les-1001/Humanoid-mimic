ACCESS
- `ssh soc` (alias already configured). Password + OTP every time, so no scripted access.
- NEVER use interactive GPU sessions (`srun --pty`, or my `soc-gpu` alias). They die silently on disconnect and keep burning fairshare. sbatch only.
- After any reconnect: `squeue -u les-1001`, and `scancel` orphaned jobs.
- Login nodes: submit and monitor only. No compute, no big pip installs, no data processing.

STORAGE
- /home/l/les-1001: tiny hidden quota (~500 MB). Never put data, venvs or checkpoints there.
- /mnt/scratch/l/les-1001: persistent storage; this project is /mnt/scratch/l/les-1001/humanoid_mimic. Probably a FILE-COUNT (inode) quota: ~37k small files once triggered "Disk quota exceeded". NEVER build a venv here; NEVER store data as many small files. Use a few hundred large shard files (WebDataset .tar). Byte limit unknown (5 GB was fine); it gets measured in feature 001.
- Compute-node /tmp: fast NVMe, but the default SLURM allocation is tiny. Always set `#SBATCH --tmp=<size>`.
- Every job: mktemp -d in local /tmp -> copy code (+ needed shards) there -> build the venv there -> run -> copy back ONLY small outputs (checkpoints, metrics, logs) to /mnt/scratch/.../artifacts/<run>/. Delete plot folders before copying back.

ENVIRONMENT
- SLURM. No `module`, no Docker, no sudo. Each job pip-installs into local /tmp.
- NEVER `. .venv/bin/activate` + bare python/pip: the inherited login environment (probably conda) hijacks PATH, so torch "installs" but fails to import. Always use VENV_PY="$(pwd)/.venv/bin/python" and VENV_PIP="$(pwd)/.venv/bin/pip".
- torch==2.14.0+cu130 worked on xgpg* nodes. MIG-sliced A100s on xgph* had drivers too old for CUDA 13 (torch.cuda.is_available() returned False).

PARTITIONS / RESOURCES
- `gpu`: max 3 h, higher priority. `gpu-long`: max 3 days. CPU work: `normal` / `long`. Max 16 running jobs per user.
- GPU jobs: `--gpus=a100-40:1` + `--nodelist=xgpg[0-7]` + `--nodes=1` (only xgpg0-7 exist; without --nodes=1 sbatch rejects the job). Pinning fixed the MIG issue; treat it as probably working until a job with it completes.
- Always set CPU and memory, or the dataloader gets OOM-killed: start with `--cpus-per-task=8 --mem=32G` (data-heavy: 16 / 64G). num_workers <= cpus-per-task.
- Anything longer than 3 days: checkpoint + resume, chained with `--dependency=afterany:<jobid>`.

PROVEN JOB TEMPLATE (keep every directive):
```sh
#!/bin/sh
#SBATCH --job-name=<name>
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --nodelist=xgpg[0-7]
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/scratch/l/les-1001/humanoid_mimic/logs/<name>_%j.out
set -e
echo "Host: $(hostname)"; df -h /tmp
WORKDIR=$(mktemp -d)
cp -r /mnt/scratch/l/les-1001/humanoid_mimic/code "$WORKDIR/"
cd "$WORKDIR/code"
python3 -m venv .venv
VENV_PY="$(pwd)/.venv/bin/python"; VENV_PIP="$(pwd)/.venv/bin/pip"
"$VENV_PIP" install -r requirements-training.txt
"$VENV_PY" -c "import torch; print('CUDA:', torch.cuda.is_available())"
"$VENV_PY" scripts/<entry>.py ...
mkdir -p /mnt/scratch/l/les-1001/humanoid_mimic/artifacts
cp -r artifacts/<name> /mnt/scratch/l/les-1001/humanoid_mimic/artifacts/
echo "Done."
```

TRANSFER RULES (my terminal mangles long pastes, especially PowerShell)
- Never ask me to paste multi-line scripts or heredocs into the remote shell. Write files locally; I scp them.
- ONE source file per scp command. A two-file scp once lost its destination in a paste and overwrote file 2 with file 1. Folders: `scp -r <folder>`.
- Short single-line commands only. Use small submit_*.sh helpers for multi-job submissions.
- The local repo is the source of truth; the cluster copy is a mirror.
