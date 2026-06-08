"""train_grpo.py — GRPO on Qwen3-1.7B for safety multiple-choice (cold, from BASE).

Dataset : cs-552-2026-vibe-trainers/mcq_safety  (safetybench + salad; A–D)
Mode    : non-thinking  (enable_thinking=False baked into the prompt)
Reward  : format + correctness  (no length / truncation term)

Direct mirror of nlp_project/multi_lin/train_grpo.py, adapted for safety and
this cluster's bitsandbytes/CUDA-12.8 quirk (Lesson 1/51).

Single A100 40 GB. Pure HuggingFace + PEFT + TRL, optional vLLM generation.

Usage (on cluster):
    cd /scratch/safety
    PYTHONPATH=/scratch/safety nohup python grpo_cold/train_grpo.py \
        > /scratch/safety/logs/grpo_cold.log 2>&1 &
    echo "PID: $!"; tail -f /scratch/safety/logs/grpo_cold.log

    python grpo_cold/train_grpo.py --config grpo_cold/cfgs_grpo_safety.yml
    python grpo_cold/train_grpo.py --vllm true     # vLLM rollouts (needs a server)

Kill condition: if `frac_reward_zero_std=1.0` from step 1 the rollouts have no
within-group variance. Cold from BASE this should NOT happen (BASE is not
overconfident); if it does, raise rollout.temperature or check the data.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import logging
import os
import sys
import time
import types
import warnings
from contextlib import contextmanager, nullcontext
from typing import Optional

# --- bitsandbytes mock (CUDA 12.8 incompatibility — Lesson 1/51) ------------
# Must run BEFORE any peft/trl import. types.ModuleType (not MagicMock) and an
# explicit __spec__ on every sub-module, or TRL's lazy-loader raises.
if "bitsandbytes" not in sys.modules:
    _bnb = types.ModuleType("bitsandbytes")
    _bnb.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", None)
    for _s in ["nn", "optim", "functional", "cextension", "cuda_setup",
               "utils", "research", "research.nn", "research.nn.modules"]:
        _sub = types.ModuleType(f"bitsandbytes.{_s}")
        _sub.__spec__ = importlib.machinery.ModuleSpec(f"bitsandbytes.{_s}", None)
        sys.modules[f"bitsandbytes.{_s}"] = _sub
        setattr(_bnb, _s.split(".")[0], _sub)
    class _Stub:  # noqa: E306
        pass
    _bnb.nn.Linear8bitLt = _Stub
    _bnb.nn.Linear4bit = _Stub
    _bnb.nn.Int8Params = _Stub
    _bnb.cextension.COMPILED_WITH_CUDA = False
    sys.modules["bitsandbytes"] = _bnb
# ---------------------------------------------------------------------------

import requests
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from format_data import SYSTEM_PROMPT, load_mc_dataset
from reward_fns import make_reward_fns

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train_grpo_safety.log")],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vLLM server management (optional)
# ---------------------------------------------------------------------------

@contextmanager
def vllm_server(cfg: DictConfig):
    vcfg = cfg.vllm
    cmd = [
        "trl", "vllm-serve",
        "--model", vcfg.model,
        "--tensor-parallel-size", str(vcfg.tensor_parallel_size),
        "--gpu-memory-utilization", str(vcfg.gpu_memory_utilization),
        "--port", str(vcfg.port),
    ]
    import subprocess

    log_path = vcfg.get("log_file", "vllm_server.log")
    timeout_s = int(vcfg.get("startup_timeout", 600))
    logf = open(log_path, "w")
    log.info("Starting vLLM server (logs -> %s): %s", log_path, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)

    health_url = f"http://localhost:{vcfg.port}/health"
    deadline = time.time() + timeout_s
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            logf.flush()
            raise RuntimeError(
                f"vLLM server exited early (code {proc.returncode}). Check '{log_path}' "
                f"— common causes: port {vcfg.port} busy, or insufficient free GPU memory."
            )
        try:
            if requests.get(health_url, timeout=2).status_code == 200:
                log.info("vLLM server is ready.")
                ready = True
                break
        except requests.ConnectionError:
            pass
        time.sleep(2)

    if not ready:
        proc.terminate()
        raise TimeoutError(
            f"vLLM server not ready within {timeout_s}s. Check '{log_path}'. "
            f"Kill stragglers: pkill -f vllm-serve"
        )

    try:
        yield f"http://localhost:{vcfg.port}/v1"
    finally:
        log.info("Shutting down vLLM server.")
        proc.terminate()
        proc.wait()
        logf.close()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_training_data(cfg: DictConfig, tokenizer) -> Dataset:
    """Build the GRPO training set."""
    d = cfg.dataset
    local = d.get("local_file")
    if local:
        import json
        recs = [json.loads(line) for line in open(local, encoding="utf-8") if line.strip()]
        base = Dataset.from_list(recs)
        log.info("Loaded %d pre-filtered prompts from %s", len(base), local)
    else:
        base = load_mc_dataset(
            sources=list(d.sources) if d.get("sources") is not None else None,
            fraction=float(d.fraction),
            seed=int(cfg.training.seed),
            hf_token=cfg.model.get("hf_token"),
        )

    def _to_prompt(row: dict) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},   # ← golden nuggets
            {"role": "user", "content": row["question_text"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,     # ← non-thinking inference
        )
        return {"prompt": prompt}

    return base.map(_to_prompt)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: DictConfig):
    tok = AutoTokenizer.from_pretrained(
        cfg.model.base_model_repo, token=cfg.model.get("hf_token"), trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model_repo,
        token=cfg.model.get("hf_token"),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map="auto",
    )

    lc = cfg.lora
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=lc.rank,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=list(lc.target_modules),
        bias="none",
    ))
    model.print_trainable_parameters()
    return model, tok


# ---------------------------------------------------------------------------
# Training args
# ---------------------------------------------------------------------------

def build_training_args(cfg: DictConfig, vllm_url: Optional[str] = None) -> GRPOConfig:
    import dataclasses

    t, a, ro = cfg.training, cfg.algorithm, cfg.rollout
    kwargs = dict(
        output_dir=t.output_dir,
        run_name=t.run_name,
        report_to=t.report_to,
        logging_steps=t.logging_steps,
        importance_sampling_level=a.importance_sampling_level,
        beta=float(a.beta),
        epsilon=float(a.epsilon),
        loss_type=a.loss_type,
        num_generations=ro.num_generations,
        max_prompt_length=ro.max_prompt_length,
        max_completion_length=ro.max_completion_length,
        temperature=float(ro.temperature),
        top_p=float(ro.top_p),
        top_k=ro.top_k,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_train_epochs=t.num_train_epochs,
        learning_rate=float(t.learning_rate),
        lr_scheduler_type=t.lr_scheduler_type,
        warmup_ratio=float(t.warmup_ratio),
        weight_decay=float(t.weight_decay),
        max_grad_norm=float(t.max_grad_norm),
        optim=t.optim,
        gradient_checkpointing=t.gradient_checkpointing,
        bf16=True,
        fp16=False,
        save_strategy=t.save_strategy,
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        seed=t.seed,
        remove_unused_columns=t.remove_unused_columns,
    )

    # Drop kwargs this installed TRL's GRPOConfig doesn't accept (version drift).
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = [k for k in kwargs if k not in valid]
    if unknown:
        log.warning("Dropping GRPOConfig kwargs unsupported by this trl: %s", unknown)
        for k in unknown:
            kwargs.pop(k)

    args = GRPOConfig(**kwargs)

    if vllm_url and cfg.vllm.enabled:
        args.use_vllm = True
        args.vllm_server_url = vllm_url
        args.vllm_model = cfg.vllm.model
        args.vllm_gpu_memory_utilization = cfg.vllm.gpu_memory_utilization
        log.info("vLLM enabled: %s", vllm_url)
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfgs_grpo_safety.yml")
    parser.add_argument("--vllm", type=lambda x: x.lower() in ("true", "1", "yes"),
                        default=None, help="Override vLLM enabled flag")
    parser.add_argument("--no_wandb", action="store_true",
                        help="disable wandb (sets report_to=none)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.vllm is not None:
        cfg.vllm.enabled = args.vllm

    if args.no_wandb or cfg.training.report_to == "none":
        cfg.training.report_to = "none"
        os.environ["WANDB_DISABLED"] = "true"
    else:
        import wandb
        wandb.init(project="grpo_safety_cold", name=cfg.training.run_name,
                   config=OmegaConf.to_container(cfg, resolve=True))

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_dataset = load_training_data(cfg, tokenizer)

    format_reward, correctness_reward = make_reward_fns(cfg)
    reward_funcs = [format_reward, correctness_reward]

    vllm_ctx = vllm_server(cfg) if cfg.vllm.enabled else nullcontext()
    with vllm_ctx as vllm_url:
        training_args = build_training_args(cfg, vllm_url)
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            reward_funcs=reward_funcs,
        )
        trainer.train()

        save_path = os.path.join(cfg.training.output_dir, "final")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.info("LoRA adapter saved to %s", save_path)

    if cfg.training.report_to != "none":
        import wandb
        wandb.finish()
    log.info("Training finished.")


if __name__ == "__main__":
    main()
