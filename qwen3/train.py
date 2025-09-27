"""
Training Qwen 3 from Scratch on Real English Data
Using publicly available datasets: FineWeb
Full production-ready implementation with proper context length and model size

torchrun --nproc_per_node=4 train.py
"""

import json
import os
import logging
import time
import random
import gc
import numpy as np
from pathlib import Path
import glob
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from contextlib import nullcontext
import itertools
from transformers import GPT2TokenizerFast
from model import Qwen3Model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

with open("config.json", "r") as f:
    QWEN3_CONFIG = json.load(f)

if isinstance(QWEN3_CONFIG["dtype"], str):
    dtype_str = QWEN3_CONFIG["dtype"]
    if dtype_str.startswith("torch."):
        dtype_name = dtype_str.split(".")[-1]
        QWEN3_CONFIG["dtype"] = getattr(torch, dtype_name)
    else:
        QWEN3_CONFIG["dtype"] = getattr(torch, dtype_str)

# Training configuration for production
TRAINING_CONFIG = {
    "batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 6e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "max_steps": 200000,  # Serious training 200000
    "eval_interval": 200,
    "save_interval": 1000,
    "max_grad_norm": 0.5,
    "dtype": torch.bfloat16,
    "compile_model": True,  # PyTorch 2.0 compilation
}

# Helper function to check if current process is main
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# ============================================================================
# Data Loading
# ============================================================================

def _load_data_shard(file):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_batch_generator(filename_pattern, batch_size, train_seq_len, bos_id=50256):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    random.shuffle(files)
    file_iter = iter(files)

    while True:
        file = next(file_iter)
        tokens = _load_data_shard(file)
        tokens_np = tokens.cpu().numpy()  # torch.uint16 -> numpy
        num_tokens = len(tokens_np)
        starts = np.arange(0, num_tokens - train_seq_len - 1, train_seq_len)
        seqs = np.lib.stride_tricks.sliding_window_view(tokens_np, train_seq_len + 1)[starts]
        batch_targets_np = seqs[:, 1:].copy()
        # 用完当前分片所有样本再切换下一个分片
        for i in range(0, len(seqs), batch_size * world_size):
            batch = seqs[i:i + batch_size * world_size]
            batch_targets = batch_targets_np[i:i + batch_size * world_size]
            if len(batch) < batch_size * world_size:
                break
            batch_inputs = torch.from_numpy(batch[:, :-1]).to(torch.int32)
            batch_targets = torch.from_numpy(batch_targets).to(torch.int64)
            batch_inputs = batch_inputs[rank::world_size]
            batch_targets = batch_targets[rank::world_size]
            yield {
                "input_ids": batch_inputs,
                "targets": batch_targets,
            }

# ============================================================================
# Enhanced Training Functions
# ============================================================================

def train_model(model, train_data_loader, val_data_loader, optimizer, scheduler, config, device, start_step=0):
    """Enhanced training function with all production features"""

    # Compile model for faster training (PyTorch 2.0+)
    if config.get("compile_model", False) and hasattr(torch, 'compile'):
        if is_main_process():
            logger.info("Compiling model for faster training...")
        model = torch.compile(model)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=(QWEN3_CONFIG["dtype"] == torch.float16))

    # Training state
    start_step = start_step
    model.train()
    step = start_step
    total_loss = 0.0
    log_interval = 100
    start_time = time.time()

    if is_main_process():
        logger.info(f"Starting training from step {start_step}")

    # Training loop
    data_iter = iter(train_data_loader)
    epoch = 0

    while step < config["max_steps"]:
        # 每200步评估一次
        if step % TRAINING_CONFIG["eval_interval"] == 0 and step > 0 and is_main_process():
            val_loss = validate_model(model, val_data_loader, device, max_batches=128, seq_len=512)
            logger.info(f"[Validation] Step {step}: avg loss={val_loss:.4f}")
    
        accum_loss = 0.0
        for micro_step in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_data_loader)
                epoch += 1
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)  # [1, seq_len]
            targets = batch["targets"].to(device, non_blocking=True)      # [1, seq_len]

            # 在梯度累积的非最后一次，关闭DDP梯度同步，减少all-reduce
            sync_ctx = (
                model.no_sync()
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                and micro_step < (config["gradient_accumulation_steps"] - 1)
                else nullcontext()
            )
            # Forward pass with mixed precision
            with sync_ctx:
                if scaler is not None and scaler.is_enabled():
                    with torch.amp.autocast(device_type=device.type, dtype=config["dtype"]):
                        _, loss = model(input_ids, targets)
                        loss = loss / config["gradient_accumulation_steps"]
                    scaler.scale(loss).backward()
                else:
                    with torch.amp.autocast(device_type=device.type, dtype=config["dtype"]):
                        _, loss = model(input_ids, targets)
                        loss = loss / config["gradient_accumulation_steps"]
                    loss.backward()

            accum_loss += float(loss.detach())

        if is_main_process():
            print(f"Step {step}: Accumulated loss: {accum_loss}")

        # Optimizer step
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad()
        step += 1
        total_loss += accum_loss

        # Logging (only main process)
        if step % log_interval == 0 and is_main_process():
            elapsed_time = time.time() - start_time
            avg_loss = total_loss / log_interval
            lr = scheduler.get_last_lr()[0]
            # 统计量：每步每卡只处理一个超长序列，token 数为 context_length
            tokens_per_sec = (
                log_interval
                * TRAINING_CONFIG["batch_size"]
                * QWEN3_CONFIG["context_length"]
                * config["gradient_accumulation_steps"]
                * (dist.get_world_size() if dist.is_initialized() else 1)
            ) / elapsed_time
            tokens_per_hour = tokens_per_sec * 3600 / 1e9

            logger.info(
                f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {lr:.4f} | "
                f"Tokens/sec: {tokens_per_sec:.0f} | Tokens/hour: {tokens_per_hour:.3f} | Epoch: {epoch}"
            )
            total_loss = 0.0
            start_time = time.time()

        # Save checkpoint (only main process)
        if step % config["save_interval"] == 0 and is_main_process():
            save_checkpoint(model, optimizer, scheduler, step, accum_loss, f"checkpoint_step_{step}.pt")

        # Memory cleanup
        if step % 1000 == 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    if is_main_process():
        logger.info("Training completed!")
    return model

def validate_model(model, val_data_loader, device, max_batches=5120, seq_len=512):
    """评估模型在验证集上的平均 loss"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        val_iter = iter(val_data_loader)
        for _ in range(max_batches):
            try:
                batch = next(val_iter)
            except StopIteration:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            # 截断到指定长度
            input_ids = input_ids[:, :seq_len]
            targets = targets[:, :seq_len]
            _, loss = model(input_ids, targets)
            # 有效 token 数（不计 ignore_index）
            valid_mask = (targets != -100)
            num_valid = valid_mask.sum().item()
            total_loss += loss.item() * num_valid
            total_tokens += num_valid

    avg_loss = total_loss / max(1, total_tokens)
    model.train()
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, step, loss, filename):
    """Enhanced checkpoint saving (only called by main process)"""
    # Extract state dict from DDP wrapper if needed
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'loss': loss,
        'config': QWEN3_CONFIG,
        'training_config': TRAINING_CONFIG
    }
    
    # Save to temporary file first, then rename (atomic operation)
    temp_filename = f"{filename}.tmp"
    torch.save(checkpoint, temp_filename)
    os.rename(temp_filename, filename)
    logger.info(f"Saved checkpoint: {filename}")

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50, device='cpu'):
    """Enhanced text generation with better sampling"""
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], device=device)
    
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                next_token_logits.scatter_(-1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end token or max length
            if generated.size(1) >= model.cfg["context_length"] - 1:
                break
            
            # Stop if we generate end token
            if next_token.item() == tokenizer.token_to_id("</s>"):
                break
    
    # Decode generated text
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline with full English data"""
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Device setup
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if is_main_process():
            print("Uses tensor cores")

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Update config with actual vocab size
    vocab_size = tokenizer.vocab_size
    QWEN3_CONFIG["vocab_size"] = vocab_size
    if is_main_process():
        logger.info(f"Final vocabulary size: {vocab_size}")

    # Load dataset
    # train_tokens_path = "fineweb10B/fineweb_train_*.bin"
    train_tokens_path = "../data/fineweb10B/fineweb_train_*.bin"
    val_tokens_path = "../data/fineweb10B/fineweb_val_*.bin"
    train_data_loader = distributed_batch_generator(
        train_tokens_path,
        batch_size=TRAINING_CONFIG["batch_size"],
        train_seq_len=QWEN3_CONFIG["context_length"]
    )
    val_data_loader = distributed_batch_generator(
        val_tokens_path,
        batch_size=1,
        train_seq_len=512,  # shorter seq len for validation
    )

    # Initialize model
    if is_main_process():
        logger.info("Initializing Qwen 3 model...")
    model = Qwen3Model(QWEN3_CONFIG)
    if is_main_process():
        print(model)
    model = model.to(device)

    # === 加载 checkpoint（如果有） ===
    checkpoint = None
    if is_main_process():
        checkpoint_files = [f for f in os.listdir(".") if f.startswith("checkpoint_step_") and f.endswith(".pt")]
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split("_")[2].split(".")[0]))
            checkpoint = latest_checkpoint
            logger.info(f"Found existing checkpoint: {checkpoint}")

    if dist.is_initialized():
        checkpoint_list = [checkpoint] if is_main_process() else [None]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]

    # === 先加载权重到未DDP的模型 ===
    if checkpoint:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if is_main_process():
            logger.info(f"Loaded checkpoint")

    # Optimizer with proper weight decay
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG["learning_rate"],
        weight_decay=TRAINING_CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        fused=True
    )
    
    # OneCycleLR scheduler with warmup
    scheduler = OneCycleLR(
        optimizer,
        max_lr=TRAINING_CONFIG["learning_rate"],
        total_steps=TRAINING_CONFIG["max_steps"],
        pct_start=TRAINING_CONFIG["warmup_steps"] / TRAINING_CONFIG["max_steps"],
        anneal_strategy='cos',
        cycle_momentum=False,
        div_factor=1e2,
        final_div_factor=1,
    )

    # Load optimizer and scheduler state if resuming
    start_step = 0
    if checkpoint:
        if is_main_process():
            logger.info("Loading optimizer and scheduler state from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']

    # Wrap model with DDP
    if dist.is_initialized():
        model = DDP(
            model, 
            device_ids=[device.index],
            find_unused_parameters=False,
            static_graph=True,
            bucket_cap_mb=50,
            gradient_as_bucket_view=True,
        )
    
    # Model info (only main process)
    if is_main_process():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 2 / 1e9:.2f} GB (bfloat16)")
    
    # Save model config (only main process)
    if is_main_process():
        with open("model_config.json", "w") as f:
            # Convert torch.dtype to string for JSON serialization
            config_for_json = QWEN3_CONFIG.copy()
            config_for_json["dtype"] = str(config_for_json["dtype"])
            json.dump(config_for_json, f, indent=2)
    
    # Train the model
    if is_main_process():
        logger.info("Starting training...")
    trained_model = train_model(
        model, 
        train_data_loader,
        val_data_loader,
        optimizer,
        scheduler,
        TRAINING_CONFIG,
        device, 
        start_step=start_step
    )
    
    # Save final model (only main process)
    if is_main_process():
        total_params = sum(p.numel() for p in trained_model.parameters())
        
        # Extract state dict from DDP wrapper if needed
        model_state_dict = trained_model.module.state_dict() if hasattr(trained_model, 'module') else trained_model.state_dict()
        
        final_checkpoint = {
            'model_state_dict': model_state_dict,
            'config': QWEN3_CONFIG,
            'training_config': TRAINING_CONFIG,
            'vocab_size': vocab_size,
            'total_params': total_params
        }
        torch.save(final_checkpoint, "final_english_qwen3.pt")
        logger.info("Final model saved!")
    
    # Test text generation (only main process)
    if is_main_process():
        logger.info("Testing text generation capabilities...")
        test_prompts = [
            "The United States is a country",
            "Machine learning is",
            "The history of artificial intelligence",
            "In the future, we will",
            "The best way to learn programming is",
            "Climate change affects",
            "The benefits of renewable energy",
            "How to improve your writing skills"
        ]
        
        # Use the model without DDP wrapper for inference
        inference_model = trained_model.module if hasattr(trained_model, 'module') else trained_model
        
        for prompt in test_prompts:
            try:
                generated = generate_text(
                    inference_model, 
                    tokenizer, 
                    prompt, 
                    max_length=100, 
                    temperature=0.7,
                    device=device
                )
                logger.info(f"Prompt: '{prompt}'")
                logger.info(f"Generated: '{generated}'\n")
            except Exception as e:
                logger.error(f"Error generating text for '{prompt}': {e}")
    
    # Save generation script (only main process)
    if is_main_process():
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved as: final_english_qwen3.pt")
        logger.info(f"Total parameters: {total_params:,}")
    
    return trained_model

if __name__ == "__main__":    
    # Run training
    main()