from __future__ import annotations

import os
from collections.abc import Iterator
from multiprocessing import Pool
from multiprocessing.pool import ApplyResult

import regex as re
from loguru import logger
from tqdm import tqdm

from cs336_basics.bpe.utils import PAT, find_chunk_boundaries


def cpu_count() -> int:
    return os.cpu_count() or 1


def pretokenize_text_iter(text: str, special_tokens: list[str]) -> Iterator[bytes]:
    if len(special_tokens) == 0:
        for pre_token in re.finditer(PAT, text):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes
        return

    start_index = 0
    # Sort special tokens by length in descending order to match longer tokens first
    sorted_tokens = sorted(special_tokens, key=len, reverse=True)
    for match in re.finditer("|".join(re.escape(token) for token in sorted_tokens), text):
        matched_token_bytes = match.group().encode("utf-8")

        for pre_token in re.finditer(PAT, text, pos=start_index, endpos=match.start()):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes

        yield matched_token_bytes
        start_index = match.end()

    if start_index < len(text):
        for pre_token in re.finditer(PAT, text, pos=start_index, endpos=len(text)):
            pre_token_bytes = pre_token.group().encode("utf-8")
            yield pre_token_bytes


def pretokenize_text(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    pretoken_counts: dict[bytes, int] = {}

    # Sort special tokens by length in descending order to match longer tokens first
    sorted_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens else []
    chunk_iter = (
        re.splititer("|".join(re.escape(token) for token in sorted_tokens), text) if len(sorted_tokens) > 0 else [text]
    )

    for chunk in chunk_iter:
        for pre_token in re.finditer(PAT, chunk):
            pre_token_bytes = pre_token.group().encode("utf-8")

            if pre_token_bytes not in pretoken_counts:
                pretoken_counts[pre_token_bytes] = 0
            pretoken_counts[pre_token_bytes] += 1

    return pretoken_counts


def pretokenize_file_chunk(
    input_path: str | os.PathLike, start: int, end: int, special_tokens: list[str]
) -> dict[bytes, int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")

    return pretokenize_text(text, special_tokens)


def merge_counts(a: dict[bytes, int], b: dict[bytes, int]) -> dict[bytes, int]:
    for key, value in b.items():
        if key not in a:
            a[key] = 0
        a[key] += value
    return a


def pretokenize_file(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[bytes, int]:
    # Reduce parallelism for large files to avoid memory exhaustion
    # Use at most 4 workers to keep memory usage reasonable
    parellel_count = min(cpu_count(), 10)
    logger.debug(
        "Starting pretokenization: input_path='{}', special_tokens={}, workers={}",
        input_path,
        special_tokens,
        parellel_count,
    )
    with open(input_path, "rb") as f:
        file_size = f.seek(0, 2)  # Seek to end to get file size
        f.seek(0)  # Seek back to beginning

        # For very large files (>1GB), use more chunks with smaller size
        # This helps with memory management
        if file_size > 1_000_000_000:  # 1GB
            num_chunks = file_size // 100_000_000
        else:
            num_chunks = parellel_count

        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        logger.debug(
            "Found {} chunk boundaries ({} chunks) for {:.2f} GB file",
            len(boundaries),
            max(0, len(boundaries) - 1),
            file_size / 1_000_000_000,
        )

    pretoken_counts: dict[bytes, int] = {}
    with Pool(parellel_count) as pool:
        # Process chunks in batches to avoid memory buildup
        chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))

        # Submit work in batches equal to worker count
        for batch_start in tqdm(range(0, len(chunk_pairs), parellel_count), desc="Processing batches"):
            batch_end = min(batch_start + parellel_count, len(chunk_pairs))
            batch_chunks = chunk_pairs[batch_start:batch_end]

            # Submit batch of work
            results: list[ApplyResult] = []
            for start, end in batch_chunks:
                results.append(pool.apply_async(pretokenize_file_chunk, (input_path, start, end, special_tokens)))

            # Process results immediately as they complete (with timeout)
            for i, result in enumerate(results):
                try:
                    counts = result.get(timeout=300)  # 5 minute timeout per chunk
                    merge_counts(pretoken_counts, counts)
                except Exception as e:
                    logger.error(f"Error processing chunk {batch_start + i}: {e}")
                    raise

    logger.debug("Pretokenization complete: {} unique pretoken strings", len(pretoken_counts))

    return pretoken_counts
