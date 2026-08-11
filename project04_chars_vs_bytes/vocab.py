"""Every GPT-2 token has two lengths. Only one of them is additive."""
import numpy as np
from transformers import AutoTokenizer

# --- the whole thing: 11 lines ---------------------------------------------
def vocab_lengths(tok):
    """Bytes and characters for every token id.

    Bytes are exact: GPT-2's surface alphabet is one printable surrogate per byte,
    so the surface length IS the byte length. Characters are not exact, because a
    character can span two tokens and decoding a token alone yields U+FFFD.
    """
    surface = tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
    n_bytes = np.array([len(s) for s in surface])
    n_chars = np.array([len(tok.decode([i])) for i in range(tok.vocab_size)])
    return n_chars, n_bytes


def bytes_per_token(text, tok, n_bytes):
    ids = tok(text)["input_ids"]
    return n_bytes[ids].sum() / len(ids)
# --- end -------------------------------------------------------------------


def load(name="gpt2"):
    return AutoTokenizer.from_pretrained(name)
