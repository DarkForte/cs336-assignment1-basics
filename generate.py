import argparse
import torch
import numpy as np

import transformer_blocks
import tokenizer

parser = argparse.ArgumentParser(description="Text generation with trained transformer")
parser.add_argument("--checkpoint", type=str, default="transformer_checkpoint_lr0.001.ckpt")
parser.add_argument("--vocab_path", type=str, default="tinystories_vocab.pickle")
parser.add_argument("--merges_path", type=str, default="tinystories_merges.pickle")
parser.add_argument("--prompt", type=str, required=True, help="Text prompt to start generation")
parser.add_argument("--max_tokens", type=int, default=1000, help="Maximum number of tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (higher = more random)")

# Model parameters (must match training configuration)
parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=4)

args = parser.parse_args()

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize model
model = transformer_blocks.Transformer(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    d_ff=args.d_ff,
    n_heads=args.n_heads,
    num_layers=args.num_layers,
    device=device
)

# Load tokenizer
tok = tokenizer.Tokenizer.from_files(args.vocab_path, args.merges_path)

# Load checkpoint
checkpoint = torch.load(args.checkpoint)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode

def generate_text(prompt: str, max_tokens: int, temperature: float = 0.8):
    # Encode the prompt
    tokens = tok.encode(prompt)
    
    # Ensure we don't exceed context length
    if len(tokens) > args.context_length:
        tokens = tokens[:args.context_length]
    
    # Convert to tensor and move to device
    context = torch.tensor(tokens).unsqueeze(0).to(device)
    
    generated = []
    for _ in range(max_tokens):
        # Get model predictions
        with torch.no_grad():
            logits = model(context) # shape: (1, context_length, vocab_size)
            
        # Focus on the last token's predictions
        next_token_logits = logits[0, -1, :] / temperature
        
        # Apply softmax to convert to probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Add to generated sequence
        generated.append(next_token.item())
        
        # Update context
        context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
        
        # If context is too long, remove first token
        if context.size(1) >= args.context_length:
            context = context[:, 1:]

        # Stop if we generate an end token (if defined)
        if tok.vocab[int(next_token.item())] == b'<|endoftext|>':
            break
    
    # Convert generated tokens to text
    full_text = prompt
    for token in generated:
        if token in tok.vocab:
            try:
                next_text = tok.vocab[token].decode('utf-8')
                full_text += next_text
            except:
                continue
    
    return full_text

# Generate text from the prompt
print("\nInput prompt:", args.prompt)
print("\nGenerated text:")
print(generate_text(args.prompt, args.max_tokens, args.temperature))
