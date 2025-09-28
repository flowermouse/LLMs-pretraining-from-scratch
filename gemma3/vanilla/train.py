"""
Training Gemma 3 from Scratch on Real English Data
Using publicly available datasets: OpenWebText
Full production-ready implementation with proper context length and model size

torchrun --nproc_per_node=4 train.py
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
import json
import os
import logging
from tokenizers import Tokenizer
import time
import numpy as np
import random
import gc
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from model import Gemma3Model
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR
from contextlib import nullcontext

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
    GEMMA3_CONFIG = json.load(f)

if isinstance(GEMMA3_CONFIG["dtype"], str):
    dtype_str = GEMMA3_CONFIG["dtype"]
    if dtype_str.startswith("torch."):
        dtype_name = dtype_str.split(".")[-1]
        GEMMA3_CONFIG["dtype"] = getattr(torch, dtype_name)
    else:
        GEMMA3_CONFIG["dtype"] = getattr(torch, dtype_str)

# Training configuration for production
TRAINING_CONFIG = {
    "batch_size": 10,  # Per device
    "gradient_accumulation_steps": 3,
    "learning_rate": 8e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "max_steps": 200000,  # Serious training 200000
    "eval_interval": 2500,
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

class EnglishDataProcessor:
    """Handles loading and processing of real English datasets with enhanced robustness"""
    
    def __init__(self, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        self.dataset_weights = {}  # For weighted sampling
        
    def load_all_datasets(self):
        """优先使用 NumPy+memmap 数据"""
        if is_main_process():
            logger.info("Loading datasets (NumPy memmap preferred)...")

        np_train_prefix = "np_data/train_tokens"
        np_val_prefix = "np_data/val_tokens"

        if is_main_process():
            logger.info("Using NumPy memmap datasets")
        self.train_tokens_mm, self.train_offsets = self._open_np(np_train_prefix)
        self.val_tokens_mm, self.val_offsets = self._open_np(np_val_prefix)

        # 用占位符维持下游接口，但实际不再构建 HuggingFace Dataset
        self.datasets = [("train_np", None), ("val_np", None)]
        self.dataset_weights["train_np"] = 0.9
        self.dataset_weights["val_np"] = 0.1

        if is_main_process():
            logger.info(f"Memmap train seq: {len(self.train_offsets)-1}, val seq: {len(self.val_offsets)-1}")

    def _open_np(self, prefix):
        offsets = np.load(f"{prefix}.offsets.npy", mmap_mode="r")
        total = int(offsets[-1])
        tokens_mm = np.memmap(f"{prefix}.tokens.int32", mode="r", dtype=np.int32, shape=(total,))
        return tokens_mm, offsets
    
    def process_datasets(self):
        """当使用 memmap 时，从 NumPy 切片流式产出 token 序列；否则沿用原来的 Dataset 迭代"""
        if not self.datasets:
            self.load_all_datasets()

        def gen_from_np(tokens_mm, offsets):
            n_seq = len(offsets) - 1
            # 可随机或顺序遍历，这里顺序遍历；如需随机可使用 np.random.permutation(n_seq)
            while True:
                for i in range(n_seq):
                    s, e = int(offsets[i]), int(offsets[i+1])
                    if e > s:
                        yield tokens_mm[s:e].tolist()

        if hasattr(self, "train_tokens_mm") and hasattr(self, "val_tokens_mm"):
            def text_generator():
                # 简单按比例交替两集
                while True:
                    for tokens in gen_from_np(self.train_tokens_mm, self.train_offsets):
                        yield tokens
                    for tokens in gen_from_np(self.val_tokens_mm, self.val_offsets):
                        yield tokens
            return text_generator()

class EnglishIterableDataset(IterableDataset):
    """Iterable dataset for streaming large English data with robustness"""
    
    def __init__(self, text_generator, tokenizer, max_length=4096, min_length=10):
        self.text_generator = text_generator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
    def __iter__(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        buffer = []
        buffer_length = 0
        sep_id = self.tokenizer.token_to_id("<sep>")

        for idx, tokens in enumerate(self.text_generator):
            # 只保留属于本 rank 的样本
            if (idx % world_size) != rank:
                continue
            try:
                if len(tokens) < self.min_length:
                    continue
                buffer.extend(tokens)
                if sep_id is not None:
                    buffer.append(sep_id)
                    buffer_length += 1
                buffer_length += len(tokens)
                while buffer_length >= self.max_length:
                    chunk = buffer[:self.max_length]
                    buffer = buffer[self.max_length//2:]
                    buffer_length = len(buffer)
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    targets = torch.tensor(chunk[1:], dtype=torch.long)
                    yield {"input_ids": input_ids, "targets": targets}
                if buffer_length > 2 * self.max_length:
                    buffer = buffer[-self.max_length:]
                    buffer_length = len(buffer)
                    gc.collect()
            except Exception as e:
                if is_main_process():
                    logger.warning(f"Error processing tokens: {e}")
                continue

# ============================================================================
# Enhanced Training Functions
# ============================================================================

def train_model(model, data_loader, optimizer, scheduler, config, device, start_step=0):
    """Enhanced training function with all production features"""

    # Compile model for faster training (PyTorch 2.0+)
    if config.get("compile_model", False) and hasattr(torch, 'compile'):
        if is_main_process():
            logger.info("Compiling model for faster training...")
        model = torch.compile(model)
    
    # Mixed precision training
    scaler = torch.amp.GradScaler(enabled=(GEMMA3_CONFIG["dtype"] == torch.float16))

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
    data_iter = iter(data_loader)
    epoch = 0
    
    while step < config["max_steps"]:
        accum_loss = 0.0
        for micro_step in range(config["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                epoch += 1
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            
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
            tokens_per_sec = (
                log_interval
                * config["batch_size"]
                * config["gradient_accumulation_steps"]
                * GEMMA3_CONFIG["context_length"]
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
        'config': GEMMA3_CONFIG,
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
    tokenizer_path = "tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if is_main_process():
        logger.info("Loaded pretrained tokenizer")
    vocab_size = tokenizer.get_vocab_size()
    if is_main_process():
        logger.info(f"Pretrained tokenizer vocab size: {vocab_size}")
    
    # Update config with actual vocab size
    actual_vocab_size = tokenizer.get_vocab_size()
    GEMMA3_CONFIG["vocab_size"] = actual_vocab_size
    if is_main_process():
        logger.info(f"Final vocabulary size: {actual_vocab_size}")
    
    # Create data processor
    if is_main_process():
        logger.info("Setting up data processor...")
    data_processor = EnglishDataProcessor(tokenizer, max_length=GEMMA3_CONFIG["context_length"])
    
    # Load datasets
    if is_main_process():
        logger.info("Loading all English datasets...")
    data_processor.load_all_datasets()
    
    if not data_processor.datasets:
        if is_main_process():
            logger.error("No datasets loaded! Cannot proceed with training.")
        return
    
    # Create streaming dataset
    if is_main_process():
        logger.info("Creating streaming dataset...")
    text_generator = data_processor.process_datasets()
    dataset = EnglishIterableDataset(
        text_generator, 
        tokenizer, 
        max_length=GEMMA3_CONFIG["context_length"]
    )

    data_loader = DataLoader(
        dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        sampler=None,
        num_workers=8,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory_device="cuda" if device.type == "cuda" else None,
    )
    
    # Initialize model
    if is_main_process():
        logger.info("Initializing Gemma 3 model...")
    model = Gemma3Model(GEMMA3_CONFIG)
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

    start_step = 0
    # Load optimizer and scheduler state if resuming
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
            config_for_json = GEMMA3_CONFIG.copy()
            config_for_json["dtype"] = str(config_for_json["dtype"])
            json.dump(config_for_json, f, indent=2)
    
    # Train the model
    if is_main_process():
        logger.info("Starting training...")
    trained_model = train_model(
        model, 
        data_loader, 
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
            'config': GEMMA3_CONFIG,
            'training_config': TRAINING_CONFIG,
            'vocab_size': actual_vocab_size,
            'total_params': total_params
        }
        torch.save(final_checkpoint, "final_english_gemma3.pt")
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
        logger.info(f"Model saved as: final_english_gemma3.pt")
        logger.info(f"Tokenizer saved as: {tokenizer_path}")
        logger.info(f"Total parameters: {total_params:,}")
    
    return trained_model

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    
    # Run training
    main()