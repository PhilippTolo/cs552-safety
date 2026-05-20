#!/usr/bin/env python3
"""Upload the repaired chat_template.jinja to HuggingFace."""
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="/scratch/safety/grpo_v5/merged/chat_template.jinja",
    path_in_repo="chat_template.jinja",
    repo_id="cs-552-2026-ma-que/safety_model",
    commit_message="Fix chat_template: user-message few-shot injection",
)
print("Uploaded chat_template.jinja")
