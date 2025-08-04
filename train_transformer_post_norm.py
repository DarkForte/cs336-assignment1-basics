import numpy as np
import torch
import train_utils
import utils
import transformer_blocks
import transformer_optimizer
import argparse
import time
import matplotlib.pyplot as plt
import signal
import sys

# Global flag for interrupt handling
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    print("\nInterrupt received. Will finish current batch and save results...")
    interrupted = True

def main():
    parser = argparse.ArgumentParser(description="Train TransformerPostNorm model")
    parser.add_argument("--total_tokens", type=int, default=40000000, help="Total tokens to process")
    parser.add_argument("--dataset_path", type=str, default="/home/darkforte/cs336/assignment-1/tinystories_encoded.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    
    # Fixed parameters as requested
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Other hyperparameters
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    # Set up interrupt handling
    signal.signal(signal.SIGINT, signal_handler)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize the TransformerPostNorm model
    transformer_model = transformer_blocks.TransformerPostNorm(
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
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    dataset = np.load(args.dataset_path, mmap_mode='r', allow_pickle=True)
    print("Data loaded successfully.")
    
    # Training loop
    tokens_processed = 0
    step = 0
    start_time = time.time()
    losses = []
    token_counts = []
    
    try:
        while tokens_processed < args.total_tokens and not interrupted:
            batch, labels = train_utils.get_batch(dataset, args.batch_size, args.context_length, device=device)
            
            optimizer.zero_grad()
            logits = transformer_model.forward(batch)
            loss = utils.cross_entropy_loss(logits, labels)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            tokens_processed += args.batch_size * args.context_length
            token_counts.append(tokens_processed)
            step += 1
            
            if step % 100 == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_processed / elapsed
                print(f"Step: {step}, Tokens: {tokens_processed:,}, Loss: {loss.item():.4f}, Time: {elapsed:.1f}s, Tokens/sec: {tokens_per_sec:.1f}")
    
    except Exception as e:
        print(f"\nError encountered: {e}")
    
    finally:
        # Always save model and plot learning curve
        final_checkpoint_path = "transformer_post_norm_final.ckpt"
        train_utils.save_checkpoint(transformer_model, optimizer, tokens_processed, final_checkpoint_path)
        print(f"\nTraining completed. Model saved to {final_checkpoint_path}")
        print(f"Total time elapsed: {time.time() - start_time:.1f}s")
        print(f"Final loss: {losses[-1]:.4f}")
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.plot(token_counts, losses, label='Training Loss')
        plt.xlabel('Tokens Processed')
        plt.ylabel('Loss')
        plt.title('Training Loss vs. Tokens Processed (Post Norm)')
        plt.grid(True)
        plt.yscale('log')
        plt.legend()
        plt.savefig('transformer_post_norm_learning_curve.png')
        plt.close()
        
        if interrupted:
            print("\nTraining interrupted by user. Results saved successfully.")
        
if __name__ == "__main__":
    main()
