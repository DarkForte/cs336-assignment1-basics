import regex as re
from typing import Iterable

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = dict([(index, value) for value, index in enumerate(merges)])
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        import pickle
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)
    
    def find_merge(self, byte_list):
        first_pair = None
        first_index = len(self.merges)
        for i in range(len(byte_list) - 1):
            pair = (byte_list[i], byte_list[i + 1])
            if pair in self.merges and self.merges[pair] < first_index:
                first_pair = pair
                first_index = self.merges[pair]
            
        return first_pair

    def encode_word(self, word: str) -> list[int]:
        byte_list = tuple(bytes([x]) for x in word.encode("utf-8"))
        while True:
            merge = self.find_merge(byte_list)
            if not merge:
                break
            new_byte_list = []
            i = 0
            while i < len(byte_list):
                if i < len(byte_list) - 1:
                    pair = (byte_list[i], byte_list[i + 1])
                    if pair == merge:
                        new_byte_list.append(pair[0] + pair[1])
                        i += 1
                    else:
                        new_byte_list.append(byte_list[i])
                else:
                    new_byte_list.append(byte_list[i])
                i += 1

            byte_list = tuple(new_byte_list)
        
        tokens = []
        for token in byte_list:
            if token in self.byte_to_id:
                tokens.append(self.byte_to_id[token])

        return tokens

    def encode(self, text) -> list[int]:
        ret = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        chunks = [text]
        if self.special_tokens:
            split_regex = "(" + "|".join([re.escape(x) for x in self.special_tokens]) + ")"
            chunks = re.split(split_regex, text)

        for chunk in chunks:
            if chunk in self.special_tokens:
                ret.append(self.byte_to_id[chunk.encode("utf-8")])
                continue
            for word in re.finditer(PAT, chunk):
                now_tokens = self.encode_word(word.group())
                if now_tokens:
                    ret.extend(now_tokens)
            
        return ret
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        ret = bytes()
        for id in ids:
            ret += self.vocab[id]
        
        return ret.decode("utf-8", errors="replace")
    

if __name__ == "__main__":
    tokenizer = Tokenizer({0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at', 11: b'<|endoftext|>', 12: b'<|endoftext|><|endoftext|>'}, 
[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')], ["<|endoftext|>", "<|endoftext|><|endoftext|>"])
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    print(re.split("!", test_string))

    print(tokenizer.encode(test_string))