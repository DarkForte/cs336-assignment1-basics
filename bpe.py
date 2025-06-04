import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import cProfile

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pre_tokenize(path: str, special_tokens: list):
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1, special_tokens[0].encode("utf-8"))
            
        # The following is a serial implementation, but you can parallelize this 
        # by sending each start/end pair to a set of processes.
        word_count = defaultdict(int)
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n")

            # Remove all special tokens
            for clean_chunk in re.split("|".join([re.escape(x) for x in special_tokens]), chunk):

                # Run pre-tokenization on your chunk and store the counts for each pre-token
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                for word in re.finditer(PAT, clean_chunk):
                #for word in re.split(" |\n", clean_chunk):
                    word_count[word.group()] += 1
    return word_count


def bpe(input_path, vocab_size, special_tokens):
    word_count = pre_tokenize(input_path, special_tokens)
    byte_count = {}
    for key, value in word_count.items():
        bytes_key = bytes(key, encoding='utf-8')
        byte_count[tuple(bytes([x]) for x in bytes_key)] = value

    merges = []
    vocab = dict([(x, bytes([x])) for x in range(0, 256)] 
                 + [(256 + i, special_token.encode("utf-8")) for i, special_token in enumerate(special_tokens)])
    p_vocab = len(vocab)

    while len(vocab) < vocab_size:
        pair_count = defaultdict(int)
        pair_to_byte_list = defaultdict(set)
        for byte_list, count in byte_count.items():
            for i in range(len(byte_list) - 1):
                pair_count[(byte_list[i], byte_list[i+1])] += count
                pair_to_byte_list[(byte_list[i], byte_list[i+1])].add(byte_list)
    
        max_count = 0
        merging_pair = tuple()
        merging_bytes = bytes()
        for key, value in pair_count.items():
            if value > max_count or (value == max_count and key > merging_pair):
                max_count = value
                merging_pair = key
                merging_bytes = merging_pair[0][:] + merging_pair[1][:]

        merges.append(merging_pair)
        vocab[p_vocab] = merging_bytes
        p_vocab += 1

        for byte_list in pair_to_byte_list[merging_pair]:
            count = byte_count[byte_list]
            del byte_count[byte_list]

            now_byte_list = []
            i = 0
            while i < len(byte_list):
                if(i + 1 < len(byte_list) and byte_list[i] == merging_pair[0] and byte_list[i+1] == merging_pair[1]):
                    now_byte_list.append(merging_bytes)
                    i += 1
                else:
                    now_byte_list.append(byte_list[i])
                i += 1
            
            byte_count[tuple(now_byte_list)] = count

    return vocab, merges

#bpe("data/small.txt", 256+6, ["<|endoftext|>"])
#bpe('tests/fixtures/tinystories_sample_5M.txt', 256+450, ['<|endoftext|>'])
bpe('/home/darkforte/cs336/assignment-1/tests/fixtures/corpus.en', 256+450, ['<|endoftext|>'])
