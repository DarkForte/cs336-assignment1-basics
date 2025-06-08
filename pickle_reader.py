import pickle
with open("vocab.pickle", "rb") as f:
    vocab = pickle.load(f)
with open("merges.pickle", "rb") as f:
    merges = pickle.load(f)
print("Vocabulary:")
for k, v in vocab.items():
    print(f"{k}: {v}")
print("\nMerges:")
for merge in merges:
    print(merge)