import os
from typing import BinaryIO
import regex as re
from collections import defaultdict
import multiprocessing as mp
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

def count_words(path, start, end, special_tokens):
    with open(path, "rb") as f:
        word_count = defaultdict(int)
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

def pre_tokenize(path: str, special_tokens: list):
    process_count = mp.cpu_count()
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, process_count, special_tokens[0].encode("utf-8"))
    
    word_count = defaultdict(int)
    with mp.Pool(process_count) as pool:
        partition_word_counts = pool.starmap(count_words, 
            [(path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])])
        # Combine the results from all partitions
        for partition in partition_word_counts:
            for word, count in partition.items():
                word_count[word] += count

    return word_count

def calc_new_byte_list(byte_list, merging_pair):
    """
    Calculate the new byte list after merging a pair of bytes.
    """
    new_byte_list = []
    i = 0
    while i < len(byte_list):
        if i < len(byte_list) - 1 and (byte_list[i], byte_list[i + 1]) == merging_pair:
            new_byte_list.append(merging_pair[0] + merging_pair[1])
            i += 2  # Skip the next byte since it's merged
        else:
            new_byte_list.append(byte_list[i])
            i += 1
    return tuple(new_byte_list)

def count_pairs(byte_list):
    """
    Count pairs of bytes in the byte list.
    Returns a dictionary with pairs as keys and their counts as values.
    """
    pair_count = defaultdict(int)
    for i in range(len(byte_list) - 1):
        pair_count[(byte_list[i], byte_list[i + 1])] += 1
    return pair_count


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

    pair_count = defaultdict(int)
    pair_to_byte_list = defaultdict(set)
    current_byte_list = {}
    for byte_list, count in byte_count.items():
        current_byte_list[byte_list] = byte_list
        for i in range(len(byte_list) - 1):
            pair_count[(byte_list[i], byte_list[i+1])] += count
            pair_to_byte_list[(byte_list[i], byte_list[i+1])].add(byte_list)

    while len(vocab) < vocab_size:
        max_count = 0
        merging_pair = tuple()
        merging_bytes = bytes()
        for key, value in pair_count.items():
            if value > max_count or (value == max_count and key > merging_pair):
                max_count = value
                merging_pair = key
                merging_bytes = merging_pair[0] + merging_pair[1]

        merges.append(merging_pair)
        vocab[p_vocab] = merging_bytes
        p_vocab += 1

        new_pair_index = defaultdict(set)
        for byte_list_raw in pair_to_byte_list[merging_pair]:
            count = byte_count[byte_list_raw]
            byte_list = current_byte_list[byte_list_raw]
            new_byte_list = calc_new_byte_list(byte_list, merging_pair)
            current_byte_list[byte_list_raw] = new_byte_list

            old_pair_count = count_pairs(byte_list)
            new_pair_count = count_pairs(new_byte_list)
            for pair, value in old_pair_count.items():
                pair_count[pair] -= value * count
            for pair, value in new_pair_count.items():
                pair_count[pair] += value * count
                new_pair_index[pair].add(byte_list_raw)
        
        del pair_to_byte_list[merging_pair]
        for pair, byte_lists in new_pair_index.items():
            if pair not in pair_to_byte_list:
                pair_to_byte_list[pair] = new_pair_index[pair]            

    return vocab, merges

if __name__ == "__main__":
    import sys
    vocab, merges = bpe(input_path=sys.argv[1], 
                        vocab_size=int(sys.argv[2]), 
                        special_tokens=["<|endoftext|>"])

    import pickle

    with open("vocab.pickle", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("merges.pickle", "wb") as f:
        pickle.dump(merges, f)

    #print(bpe("data/small.txt", 10000, ["<|endoftext|>"]))
    #bpe('tests/fixtures/tinystories_sample_5M.txt', 256+450, ['<|endoftext|>'])
    #bpe('/home/darkforte/cs336/assignment-1/tests/fixtures/corpus.en', 256+450, ['<|endoftext|>'])
