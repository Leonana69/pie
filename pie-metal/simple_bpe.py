"""
Simple BPE tokenizer for testing - Python port of backend/backend-cuda/src/bpe.cpp
"""

import re
import base64
from typing import Dict, List, Set


def load_merge_rules(vocab_file_path: str) -> Dict[bytes, int]:
    """
    Load merge rules from vocabulary file.
    Format: base64_token rank
    """
    merge_rules = {}
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Line {line_num}: expected 2 parts, got {len(parts)}")

            b64_token, rank_str = parts
            try:
                decoded_token = base64.b64decode(b64_token)
                rank = int(rank_str)
                merge_rules[decoded_token] = rank
            except Exception as e:
                raise ValueError(f"Line {line_num}: error decoding - {e}")

    return merge_rules


def byte_pair_merge(piece: bytes, ranks: Dict[bytes, int]) -> List[int]:
    """Apply byte-pair merging to a piece."""
    if len(piece) == 0:
        return []

    if len(piece) == 1:
        if piece in ranks:
            return [ranks[piece]]
        raise RuntimeError(f"Single byte token not found: {piece}")

    # Initialize parts: each position is a potential split point
    parts = [(i, float('inf')) for i in range(len(piece) + 1)]

    def get_rank_for_pair(start: int, end: int) -> float:
        if start + 1 >= end:
            return float('inf')
        sub = piece[start:end]
        return ranks.get(sub, float('inf'))

    # Initialize ranks between adjacent positions
    for i in range(len(parts) - 2):
        parts[i] = (parts[i][0], get_rank_for_pair(parts[i][0], parts[i + 1][0]))

    # Merge pairs with lowest rank iteratively
    while len(parts) > 1:
        # Find minimum rank
        min_rank = float('inf')
        min_idx = -1
        for i in range(len(parts) - 1):
            if parts[i][1] < min_rank:
                min_rank = parts[i][1]
                min_idx = i

        if min_rank == float('inf'):
            break

        # Merge the best pair by removing the split point
        parts.pop(min_idx + 1)

        # Update ranks around the merge
        if min_idx > 0:
            parts[min_idx - 1] = (parts[min_idx - 1][0],
                                  get_rank_for_pair(parts[min_idx - 1][0], parts[min_idx][0]))
        if min_idx < len(parts) - 1:
            parts[min_idx] = (parts[min_idx][0],
                            get_rank_for_pair(parts[min_idx][0], parts[min_idx + 1][0]))

    # Convert parts to tokens
    tokens = []
    for i in range(len(parts) - 1):
        sub_piece = piece[parts[i][0]:parts[i + 1][0]]
        if sub_piece not in ranks:
            raise RuntimeError(f"Token not found after merge: {sub_piece}")
        tokens.append(ranks[sub_piece])

    return tokens


class SimpleBPETokenizer:
    """Simple BPE tokenizer compatible with LLaMA 3."""

    def __init__(self, vocab_path: str, special_tokens: Dict[str, int] = None):
        """Load tokenizer from vocabulary file with special tokens from model config."""
        self.encoder = load_merge_rules(vocab_path)

        # Use provided special tokens or defaults for LLaMA 3
        if special_tokens is None:
            self.special_tokens = {
                "<|begin_of_text|>": 128000,
                "<|end_of_text|>": 128001,
                "<|start_header_id|>": 128006,
                "<|end_header_id|>": 128007,
                "<|eot_id|>": 128009,
            }
        else:
            self.special_tokens = special_tokens

        # LLaMA 3 split pattern (ASCII-based for portability)
        self.pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d|[^\r\na-z0-9]?[a-z]+|[0-9]{1,3}| ?[^\sa-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            re.IGNORECASE
        )

        # Build special token regex
        special_pattern_parts = []
        for token in self.special_tokens.keys():
            # Escape regex special characters
            escaped = re.escape(token)
            special_pattern_parts.append(escaped)

        self.special_pattern = re.compile('|'.join(special_pattern_parts))

        # Build decoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        for token_str, token_id in self.special_tokens.items():
            self.decoder[token_id] = token_str.encode('utf-8')

    def encode(self, text: str, allowed_special: Set[str] = None) -> List[int]:
        """Encode text to token IDs."""
        if allowed_special is None:
            allowed_special = set()

        tokens = []

        def process_chunk(chunk_text: str):
            """Process a text chunk (no special tokens)."""
            if not chunk_text:
                return

            # Split by pattern
            for match in self.pattern.finditer(chunk_text):
                piece = match.group().encode('utf-8')

                # Check if piece is directly in encoder
                if piece in self.encoder:
                    tokens.append(self.encoder[piece])
                else:
                    # Apply BPE merging
                    piece_tokens = byte_pair_merge(piece, self.encoder)
                    tokens.extend(piece_tokens)

        if not allowed_special:
            # No special tokens to handle
            process_chunk(text)
            return tokens

        # Handle special tokens
        last_pos = 0
        for match in self.special_pattern.finditer(text):
            special_token = match.group()

            if special_token in allowed_special:
                # Process text before this special token
                process_chunk(text[last_pos:match.start()])

                # Add special token
                tokens.append(self.special_tokens[special_token])

                # Update position
                last_pos = match.end()

        # Process remaining text
        process_chunk(text[last_pos:])

        return tokens

    def encode_with_special_tokens(self, text: str) -> List[int]:
        """Encode text allowing all special tokens."""
        return self.encode(text, allowed_special=set(self.special_tokens.keys()))

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        decoded_bytes = bytearray()
        for token_id in tokens:
            if token_id not in self.decoder:
                raise ValueError(f"Unknown token ID: {token_id}")
            decoded_bytes.extend(self.decoder[token_id])

        return decoded_bytes.decode('utf-8', errors='replace')
