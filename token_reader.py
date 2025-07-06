import numpy as np
from tokenizer import Tokenizer

encoded = np.load("tinystories_encoded.npy", mmap_mode='r')
print("Encoded data loaded successfully.")
print("encoded tokens (sample): ", encoded[:100])
tokenizer = Tokenizer.from_files("tinystories_vocab.pickle", "tinystories_merges.pickle", special_tokens=["<|endoftext|>"])
decoded = tokenizer.decode(encoded[:1000].tolist())
print("Decoded data:", decoded[:1000])  # Print first 100 characters of the decoded data