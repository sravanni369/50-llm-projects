"""Part 1: the book exercise — three schemes on one sentence, with comparison plots."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from schemes import char_tokenize, word_tokenize, gpt2_tokenize, gpt2_tokenizer

TXT = 'The way you do anything is the way you do everything.'


def token_scatter(tokens, vocab, xlabel, fname):
    _, ax = plt.subplots(1, figsize=(12, 4))
    ax.plot(tokens, 'ks', markersize=12, markerfacecolor=[.7, .7, .9])
    ax.set(xlabel=xlabel, yticks=range(len(vocab)),
           yticklabels=[repr(v)[1:-1] for v in vocab])
    ax.grid(linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    char_toks, char_vocab = char_tokenize(TXT)
    print(f'Characters: {len(char_toks)} total, {len(char_vocab)} unique')
    print('char vocab:', char_vocab)
    token_scatter(char_toks, char_vocab, 'Character index', 'part1_char_tokens.png')

    word_toks, word_vocab = word_tokenize(TXT)
    print(f'Words: {len(word_toks)} total, {len(word_vocab)} unique')
    print('word vocab:', word_vocab)
    token_scatter(word_toks, word_vocab, 'Word index', 'part1_word_tokens.png')

    gpt2_toks = gpt2_tokenize(TXT)
    tok = gpt2_tokenizer()
    print(f'GPT-2: {len(gpt2_toks)} total, {len(set(gpt2_toks))} unique')
    for t in gpt2_toks:
        print(f'  token {t:5} = {tok.decode([t])!r}')

    # leading-space effect: same word, different token
    for w in [' Mike', 'Mike']:
        print(f'GPT-2 tokens for {w!r}: {tok.encode(w)}')

    schemes = ['Characters', 'Words', 'GPT-2']
    totals = [len(char_toks), len(word_toks), len(gpt2_toks)]
    uniques = [len(char_vocab), len(word_vocab), len(set(gpt2_toks))]

    _, axs = plt.subplots(1, 2, figsize=(10, 3))
    for i in range(3):
        axs[0].bar(i, totals[i])
        axs[1].bar(i, uniques[i])
    axs[0].set(xticks=range(3), xticklabels=schemes, ylabel='Count', title='Total tokens')
    axs[1].set(xticks=range(3), xticklabels=schemes, ylabel='Count', title='Unique tokens')
    plt.tight_layout()
    plt.savefig('part1_comparison.png')
    plt.close()
    print('saved: part1_char_tokens.png, part1_word_tokens.png, part1_comparison.png')


if __name__ == '__main__':
    main()
