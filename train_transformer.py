import sys
import numpy as np
import torch

import transformer_blocks
import transformer_optimizer
import train_utils

import argparse

parser = argparse.ArgumentParser(description="Transformer model arguments")
parser.add_argument("--vocab_size", type=int, required=True)
parser.add_argument("--context_length", type=int, required=True)
parser.add_argument("--d_model", type=int, required=True)
parser.add_argument("--d_ff", type=int, required=True)
parser.add_argument("--n_heads", type=int, required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--context_length", type=int, required=True)

args = parser.parse_args()

