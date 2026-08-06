import pytest

from schemes import (char_tokenize, char_decode, word_tokenize, word_decode,
                     gpt2_tokenize, gpt2_tokenizer, scheme_counts)

TXT = 'The way you do anything is the way you do everything.'


def test_char_roundtrip():
    tokens, vocab = char_tokenize(TXT)
    assert char_decode(tokens, vocab) == TXT
    assert len(tokens) == len(TXT)


def test_char_vocab_sorted_unique():
    _, vocab = char_tokenize(TXT)
    assert vocab == sorted(set(TXT))
    assert len(vocab) == len(set(vocab))


def test_word_roundtrip():
    tokens, vocab = word_tokenize(TXT)
    assert word_decode(tokens, vocab) == TXT
    assert len(tokens) == 11


def test_word_case_sensitive():
    # 'The' and 'the' are different word tokens
    _, vocab = word_tokenize(TXT)
    assert 'The' in vocab and 'the' in vocab


def test_gpt2_roundtrip():
    tok = gpt2_tokenizer()
    assert tok.decode(gpt2_tokenize(TXT)) == TXT


def test_gpt2_leading_space_changes_token():
    assert gpt2_tokenize(' Mike') != gpt2_tokenize('Mike')


def test_scheme_counts_ordering():
    # BPE compresses: fewer tokens than characters, more than (or equal to) words
    c = scheme_counts(TXT)
    assert c['word']['total'] <= c['gpt2']['total'] < c['char']['total']


def test_skill_term_fragmentation():
    tok = gpt2_tokenizer()
    assert len(tok.encode(' SQL')) == 1
    assert len(tok.encode(' PyTorch')) == 3
    assert len(tok.encode(' scikit-learn')) == 5
    assert len(tok.encode('Excel')) == 2  # no leading space: fragments


def test_telugu_costs_more():
    en = 'Children need good nutrition to grow well.'
    te = 'పిల్లలు బాగా ఎదగడానికి మంచి పోషకాహారం అవసరం.'
    assert len(gpt2_tokenize(te)) > 2 * len(gpt2_tokenize(en))
