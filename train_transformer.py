import sys
import numpy as np
import torch

import transformer_blocks
import transformer_optimizer
import train_utils
import utils

import argparse

parser = argparse.ArgumentParser(description="Transformer model arguments")
parser.add_argument("--dataset_path", type=str, default="/home/darkforte/cs336/assignment-1/tinystories_encoded.npy")

parser.add_argument("--vocab_size", type=int, default=10000)
parser.add_argument("--context_length", type=int, default=256)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_ff", type=int, default=1344)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--num_layers", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.95)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--total_tokens", type=int, default=10000)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

transformer_model = transformer_blocks.Transformer(vocab_size=args.vocab_size,
                               context_length=args.context_length,
                               d_model=args.d_model,
                               d_ff=args.d_ff,
                               n_heads=args.n_heads,
                               num_layers=args.num_layers,
                               device=device)

optimizer = transformer_optimizer.AdamW(transformer_model.parameters(),
                                        lr=args.lr,
                                        betas=(args.beta1, args.beta2),
                                        weight_decay=args.weight_decay,
                                        eps=1e-8)

dataset = np.load(args.dataset_path, mmap_mode='r')
print("Data loaded successfully.")

now_tokens = 0
while now_tokens < args.total_tokens:
    batch, labels = train_utils.get_batch(dataset, args.batch_size, args.context_length, device=device)
    print(batch)
    
    optimizer.zero_grad()
    logits = transformer_model.forward(batch)

    loss = utils.cross_entropy_loss(logits, labels)  # Assuming 0 is the padding index
    loss.backward()
    
    optimizer.step()
    
    now_tokens += args.batch_size * args.context_length
    print(f"Processed {now_tokens} tokens, current loss: {loss.item()}")
    
    if now_tokens % 1000 == 0:
        print(f"Processed {now_tokens} tokens, current loss: {loss.item()}")