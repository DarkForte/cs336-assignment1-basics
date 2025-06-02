import regex as re

print(re.split("|".join([re.escape("<|endoftext|>")]), "aaaabs<|endoftext|>ppp"))