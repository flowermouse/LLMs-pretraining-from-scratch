# Pretraining

This directory contains code for pretraining language models, including Gemma3 and Qwen3 series models.

# Architecture

![Architecture Diagram](./asset/gemma3-vs-qwen3.webp)

# Acknowledgements
This repository references the following repositories:
- LLMs from scratch
- Gemma3 from scratch
- Modded nanoGPT

# Features
## Speed up training
### Model
- Flash Attention
    - PyTorch SPDA `F.scaled_dot_product_attention`
        - Only support no mask or causal mask
    - Flex Attention
        - Support any self designed mask
        - Easy to implement, no need to wright kernel code
- FFN: Merged two parallel projections into a single GEMM operation that outputs `2 * hidden_dim`, followed by chunking.  
- QKV: Merged Q, K, V projections into a single GEMM operation that outputs `3 * hidden_dim`, followed by chunking.
### Model settings that influence speed
- Use `bfloat16`
- max_seq_len = 512 will be faster than 1024
- Sparse attention will be faster than dense attention
    - Gemma3: use `sliding_window_attention : global_attention = 5 : 1`
- A large vocab size will slow down training
    - GPT2 tokenizer vocab size is 50257
    - Llama tokenizer vocab size is 32000
    - Qwen tokenizer vocab size is 151936
    - Gemma3 tokenizer vocab size is 262145

### Tips
- Use `torch.compile` to speed up training
- Use Fused AdamW optimizer
- Set `torch.backends.cuda.matmul.allow_tf32 = True` to speed up training
- Set `torch.backends.cudnn.benchmark = True` to speed up training
- Set `PYTORCH_ALLOC_CONF="expandable_segments:True"` to speed up training
- Set `torch.set_float32_matmul_precision("high")` to use tensor cores

### Note
- Need no fancy propressing on the training batch, just concatenate all text files into a single file, then read.
- Gemma3 official model parameter size is 270M, but by definition it is about 430M, the difference is Gemma3 uses shared weights for the embedding layer and the output layer. So the vanilla version should also use shared weights.
