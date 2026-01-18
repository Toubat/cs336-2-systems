from cs336_basics.bpe.utils import find_chunk_boundaries

with open("data/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
    num_processes = 400
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        print(chunk)
        print("=" * 100)
