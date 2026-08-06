"""Three tokenization schemes: character, word, GPT-2 BPE.

Own implementation of Project 1 from "50 ML Projects To Understand LLMs"
(Mike X Cohen, MIT-licensed companion code: github.com/mikexcohen/ML4LLM_book).
"""

from functools import lru_cache


def char_tokenize(text):
    """Encode text as indices into its sorted unique-character vocab."""
    vocab = sorted(set(text))
    tokens = [vocab.index(c) for c in text]
    return tokens, vocab


def char_decode(tokens, vocab):
    return ''.join(vocab[t] for t in tokens)


def word_tokenize(text):
    """Encode whitespace-split words as indices into their sorted unique vocab."""
    words = text.split()
    vocab = sorted(set(words))
    tokens = [vocab.index(w) for w in words]
    return tokens, vocab


def word_decode(tokens, vocab):
    return ' '.join(vocab[t] for t in tokens)


@lru_cache(maxsize=1)
def gpt2_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('gpt2')


def gpt2_tokenize(text):
    return gpt2_tokenizer().encode(text)


def scheme_counts(text):
    """Total and unique token counts for all three schemes on one text."""
    char_toks, _ = char_tokenize(text)
    word_toks, _ = word_tokenize(text)
    gpt2_toks = gpt2_tokenize(text)
    return {
        'char': {'total': len(char_toks), 'unique': len(set(char_toks))},
        'word': {'total': len(word_toks), 'unique': len(set(word_toks))},
        'gpt2': {'total': len(gpt2_toks), 'unique': len(set(gpt2_toks))},
    }
