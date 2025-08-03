import matplotlib.pyplot as plt
import numpy as np
import torch
import train_utils
import utils
import transformer_blocks
import transformer_optimizer
import argparse
import time

def train_with_lr(lr, args, dataset, total_tokens):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    transformer_model = transformer_blocks.Transformer(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        num_layers=args.num_layers,
        device=device
    )
    
    optimizer = transformer_optimizer.AdamW(
        transformer_model.parameters(),
        lr=lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    losses = []
    tokens_processed = 0
    step = 0
    start_time = time.time()
    
    while tokens_processed < total_tokens:
        batch, labels = train_utils.get_batch(dataset, args.batch_size, args.context_length, device=device)
        
        optimizer.zero_grad()
        logits = transformer_model.forward(batch)
        loss = utils.cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        tokens_processed += args.batch_size * args.context_length
        step += 1
        
        if step % 100 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            print(f"LR: {lr}, Tokens: {tokens_processed:,}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s, Tokens/sec: {tokens_per_sec:.1f}")
    
    train_utils.save_checkpoint(transformer_model, optimizer, tokens_processed, f"transformer_checkpoint_lr{lr}.ckpt")
    return losses

def main():
    parser = argparse.ArgumentParser(description="Learning rate sweep for Transformer")
    parser.add_argument("--total_tokens", type=int, default=40000000, help="Total tokens to process for each learning rate")
    parser.add_argument("--dataset_path", type=str, default="/home/darkforte/cs336/assignment-1/tinystories_encoded.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Define learning rates to try
    learning_rates = [1e-4, 1e-3, 1e-2]
    
    dataset = np.load(args.dataset_path, mmap_mode='r', allow_pickle=True)
    print("Data loaded successfully.")
    
    # Run training for each learning rate
    all_losses = {}
    for lr in learning_rates:
        print(f"\nTraining with learning rate: {lr}")
        losses = train_with_lr(lr, args, dataset, args.total_tokens)
        all_losses[lr] = losses
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for lr, losses in all_losses.items():
        plt.plot(losses, label=f'lr={lr}')
    
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps for Different Learning Rates')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('lr_sweep.png')
    plt.close()

if __name__ == "__main__":
    main()
