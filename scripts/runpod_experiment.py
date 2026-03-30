#!/usr/bin/env python3
"""Automate S2 experiment on RunPod via pexpect SSH."""

import pexpect
import sys
import time
import os

SSH_CMD = (
    "ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no "
    "-o UserKnownHostsFile=/dev/null "
    "n29eyv6mwshgyg-64410ecc@ssh.runpod.io "
    "-i /home/vir1/.ssh/id_ed25519"
)

PROMPT = r'root@[a-z0-9]+:.+[#$]'
BASE = "/workspace/S2-conformal-uq"

# Files to transfer: (local_path, remote_path)
PROJECT_ROOT = "/home/vir1/.openclaw/workspace/neurips2026/S2-conformal-uq"
FILES = [
    ("src/__init__.py", f"{BASE}/src/__init__.py"),
    ("src/utils.py", f"{BASE}/src/utils.py"),
    ("src/models.py", f"{BASE}/src/models.py"),
    ("src/data.py", f"{BASE}/src/data.py"),
    ("src/scores.py", f"{BASE}/src/scores.py"),
    ("src/conformal.py", f"{BASE}/src/conformal.py"),
    ("src/evaluation.py", f"{BASE}/src/evaluation.py"),
    ("scripts/train.py", f"{BASE}/scripts/train.py"),
    ("scripts/calibrate.py", f"{BASE}/scripts/calibrate.py"),
]


def run_cmd(child, cmd, timeout=30, show=True):
    """Send a command and wait for prompt, return output."""
    child.sendline(cmd)
    child.expect(PROMPT, timeout=timeout)
    output = child.before.decode("utf-8", errors="replace")
    # Strip ANSI escape codes
    import re
    output = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
    output = re.sub(r'\x1b\][^\x07]*\x07', '', output)
    output = re.sub(r'\[[\?]?[0-9]*[a-z]', '', output)
    if show:
        # Print only the relevant part (skip the echoed command)
        lines = output.strip().split('\n')
        if len(lines) > 1:
            for line in lines[1:]:
                stripped = line.strip()
                if stripped:
                    print(stripped)
    return output


def transfer_file(child, local_rel, remote_path):
    """Transfer a file via base64 encoding over SSH."""
    local_path = os.path.join(PROJECT_ROOT, local_rel)
    with open(local_path, 'r') as f:
        content = f.read()

    # Create directory
    remote_dir = os.path.dirname(remote_path)
    run_cmd(child, f"mkdir -p {remote_dir}", show=False)

    # Handle empty files
    if not content.strip():
        run_cmd(child, f"touch {remote_path}", timeout=10, show=False)
        print(f"  {local_rel} -> {remote_path} [OK (empty)]")
        return

    # Use base64 to avoid any shell escaping issues
    import base64
    encoded = base64.b64encode(content.encode()).decode()

    # Transfer in chunks to avoid line length limits
    chunk_size = 500
    chunks = [encoded[i:i+chunk_size] for i in range(0, len(encoded), chunk_size)]

    # Write first chunk (overwrite)
    run_cmd(child, f"echo -n '{chunks[0]}' > /tmp/_transfer.b64", timeout=10, show=False)
    # Append remaining chunks
    for chunk in chunks[1:]:
        run_cmd(child, f"echo -n '{chunk}' >> /tmp/_transfer.b64", timeout=10, show=False)

    # Decode and write
    run_cmd(child, f"base64 -d /tmp/_transfer.b64 > {remote_path}", timeout=10, show=False)

    # Verify
    output = run_cmd(child, f"wc -c < {remote_path}", show=False)
    import re
    nums = re.findall(r'\d+', output)
    remote_size = int(nums[-1]) if nums else 0
    local_size = len(content.encode())
    status = "OK" if abs(remote_size - local_size) <= 1 else f"MISMATCH (local={local_size}, remote={remote_size})"
    print(f"  {local_rel} -> {remote_path} [{status}]")


def main():
    print("=" * 60)
    print("S2 RunPod GPU Experiment")
    print("=" * 60)

    # Connect
    print("\n[1/6] Connecting to RunPod...")
    child = pexpect.spawn(SSH_CMD, timeout=30, encoding=None)
    child.logfile_read = None  # suppress raw output

    # Wait for shell prompt
    child.expect(PROMPT, timeout=30)
    print("Connected!")

    # Check GPU
    print("\n[2/6] Checking GPU...")
    run_cmd(child, "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
    run_cmd(child, "python3 --version")

    # Create directory structure
    print("\n[3/6] Transferring files...")
    run_cmd(child, f"mkdir -p {BASE}/src {BASE}/scripts {BASE}/checkpoints {BASE}/results", show=False)

    for local_rel, remote_path in FILES:
        transfer_file(child, local_rel, remote_path)

    # Verify structure
    print("\nRemote file structure:")
    run_cmd(child, f"find {BASE} -type f | sort")

    # Install dependencies
    print("\n[4/6] Installing dependencies...")
    run_cmd(child, "pip install neuraloperator torch h5py numpy 2>&1 | tail -5", timeout=300)
    run_cmd(child, "python3 -c 'import neuralop; print(\"neuralop OK:\", neuralop.__version__)'", timeout=30)
    run_cmd(child, "python3 -c 'import torch; print(\"torch OK:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())'", timeout=30)

    # Run training
    print("\n[5/6] Training FNO on Darcy (500 epochs)...")
    print("This will take several minutes on GPU...")
    train_cmd = (
        f"cd {BASE} && python3 scripts/train.py "
        "--model fno --pde darcy --resolution 64 "
        "--data_source neuralop --epochs 500 --lr 1e-3 "
        "--n_train 800 --n_cal 100 --n_test 100 "
        "--save_dir ./checkpoints 2>&1"
    )
    run_cmd(child, train_cmd, timeout=600)

    # Run calibration
    print("\n[6/6] Running conformal calibration...")
    cal_cmd = (
        f"cd {BASE} && python3 scripts/calibrate.py "
        "--model fno --pde darcy --resolution 64 "
        "--data_source neuralop --checkpoint_dir ./checkpoints "
        "--output_dir ./results --mc_dropout "
        "--n_train 800 --n_cal 100 --n_test 100 2>&1"
    )
    run_cmd(child, cal_cmd, timeout=300)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    run_cmd(child, f"cat {BASE}/results/fno_darcy_64/calibration_results.json")

    print("\n--- TSV Log ---")
    run_cmd(child, f"cat {BASE}/results/results.tsv")

    # Also get checkpoint info
    print("\n--- Checkpoint Info ---")
    run_cmd(child, f"python3 -c \"import torch; c=torch.load('{BASE}/checkpoints/fno_darcy_64/best.pt', map_location='cpu', weights_only=False); print('Best epoch:', c['epoch'], 'Loss:', c['loss'])\"", timeout=30)

    child.sendline("exit")
    child.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
