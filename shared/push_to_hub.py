"""
Upload a merged safety checkpoint to HuggingFace Hub.

Reads upcfgs.yml (at the repo root) for paths and repo ID, then calls
HfApi.upload_folder — no model is loaded into memory.

Run merge_lora.py first so the merged directory already contains
generation_config.json and the patched chat_template.jinja.

Usage (from cs552-safety/ root):
    python shared/push_to_hub.py

Or with explicit overrides:
    python shared/push_to_hub.py --checkpoint /scratch/safety/sft_vibe4/merged \
                                  --repo-id cs-552-2026-ma-que/safety_model
"""

import argparse
import os
import sys

from dotenv import load_dotenv
from huggingface_hub import HfApi
from omegaconf import OmegaConf

load_dotenv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Local merged model path (overrides upcfgs.yml output_path)")
    p.add_argument("--repo-id", default=None,
                   help="HuggingFace repo, e.g. cs-552-2026-ma-que/safety_model "
                        "(overrides upcfgs.yml repo_id)")
    p.add_argument("--config", default=None,
                   help="Path to upcfgs.yml (default: upcfgs.yml next to this script's cwd)")
    args = p.parse_args()

    # Locate upcfgs.yml relative to script or cwd
    cfg_path = args.config
    if cfg_path is None:
        for candidate in [
            os.path.join(os.path.dirname(__file__), "..", "upcfgs.yml"),
            "upcfgs.yml",
        ]:
            if os.path.exists(candidate):
                cfg_path = candidate
                break

    cfg = OmegaConf.load(cfg_path) if cfg_path and os.path.exists(cfg_path) else None

    checkpoint = args.checkpoint or (cfg.output_path if cfg else None)
    repo_id    = args.repo_id    or (cfg.repo_id    if cfg else None)

    if not checkpoint:
        sys.exit("ERROR: no checkpoint path — set output_path in upcfgs.yml or pass --checkpoint")
    if not repo_id:
        sys.exit("ERROR: no repo ID — set repo_id in upcfgs.yml or pass --repo-id")

    print(f"Uploading  : {checkpoint}")
    print(f"→ Repo     : {repo_id}")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=checkpoint, repo_id=repo_id, repo_type="model")

    print(f"\nDone! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
