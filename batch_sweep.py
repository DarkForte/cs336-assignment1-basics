import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from sweep_utils import train_with_params

def main():
    parser = argparse.ArgumentParser(description="Batch size sweep for Transformer")
    parser.add_argument("--dataset_path", type=str, default="/home/darkforte/cs336/assignment-1/tinystories_encoded.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    # Define batch sizes to try
    batch_sizes = [1, 4, 8, 32, 64, 128]
    
    dataset = np.load(args.dataset_path, mmap_mode='r', allow_pickle=True)
    print("Data loaded successfully.")
    
    # Run training for each batch size
    all_losses = {}
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        losses = train_with_params(args, dataset, min(batch_size * args.context_length * 20000, 40000000), "batch_size", batch_size)
        all_losses[batch_size] = losses
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for batch_size, losses in all_losses.items():
        plt.plot(losses, label=f'batch_size={batch_size}')
    
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps for Different Batch Sizes')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('batch_size_sweep.png')
    plt.close()

if __name__ == "__main__":
    main()
