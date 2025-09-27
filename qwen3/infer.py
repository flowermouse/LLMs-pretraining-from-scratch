"""
Text generation script for trained English Qwen 3 model
"""
import torch
from transformers import GPT2TokenizerFast
from model import Qwen3Model
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=128, temperature=1.0, top_k=20, device='cpu'):
    """Enhanced text generation with better sampling"""
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
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
            next_token = next_token.unsqueeze(1)      # shape: (1, 1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end token or max length
            if generated.size(1) >= model.cfg["context_length"] - 1:
                break

            # Stop if we generate end token (GPT2 does not have </s>, use 50256)
            eos_token_id = 50256  # GPT-2 的 <|endoftext|> id
            if next_token.item() == eos_token_id:
                break

    # Decode generated text
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text

def load_model():
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # import tiktoken
    # tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load model
    checkpoint = torch.load("checkpoint_step_166000.pt", map_location="cpu")
    
    # Recreate model
    model = Qwen3Model(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        emb = model.tok_emb.weight
        out = model.out_head.weight
        print("Embedding mean/std:", emb.mean().item(), emb.std().item())
        print("Output head mean/std:", out.mean().item(), out.std().item())
    
    return model, tokenizer

if __name__ == "__main__":
    model, tokenizer = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    while True:
        prompt = input("Enter prompt (or 'quit'): ")
        if prompt.lower() == 'quit':
            break
        
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        generated = generate_text(model, tokenizer, prompt, device=device)
        print(f"Generated: {generated}")