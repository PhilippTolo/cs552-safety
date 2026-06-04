"""Training utilities shared across SFT and GRPO scripts."""

import re
from pathlib import Path


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the most recent checkpoint-{N} folder, or None."""
    ckpt_dir = Path(output_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = []
    for p in ckpt_dir.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        return None
    step, path = max(checkpoints, key=lambda x: x[0])
    print(f"[Resume] Found checkpoint at step {step}: {path}")
    return str(path)
