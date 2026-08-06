"""Part 2: what does Telugu cost in GPT-2 tokens vs English?

Same meaning, two languages, three schemes. The headline metric is
fertility (tokens per word) and the Telugu/English token-count ratio
under GPT-2's byte-level BPE, which has almost no Telugu merges.
"""

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from schemes import char_tokenize, word_tokenize, gpt2_tokenize

# stock Windows console is cp1252 and cannot print Telugu
sys.stdout.reconfigure(encoding='utf-8')

# translation pairs (author-written, same meaning in both languages)
PAIRS = [
    ('The government announced a new scheme for farmers today.',
     'ప్రభుత్వం రైతుల కోసం ఈరోజు కొత్త పథకం ప్రకటించింది.'),
    ('Children need good nutrition to grow well.',
     'పిల్లలు బాగా ఎదగడానికి మంచి పోషకాహారం అవసరం.'),
    ('Education changes the lives of women.',
     'విద్య మహిళల జీవితాలను మారుస్తుంది.'),
]


def sentence_stats(text):
    n_words = len(text.split())
    n_gpt2 = len(gpt2_tokenize(text))
    return {
        'chars': len(char_tokenize(text)[0]),
        'words': n_words,
        'gpt2': n_gpt2,
        'fertility': n_gpt2 / n_words,
    }


def main():
    en_totals, te_totals = [], []
    for en, te in PAIRS:
        s_en, s_te = sentence_stats(en), sentence_stats(te)
        en_totals.append(s_en['gpt2'])
        te_totals.append(s_te['gpt2'])
        print(f'EN: {en}')
        print(f"    {s_en['chars']} chars | {s_en['words']} words | "
              f"{s_en['gpt2']} GPT-2 tokens | fertility {s_en['fertility']:.1f}")
        print(f'TE: {te}')
        print(f"    {s_te['chars']} chars | {s_te['words']} words | "
              f"{s_te['gpt2']} GPT-2 tokens | fertility {s_te['fertility']:.1f}")
        print(f"    Telugu/English token ratio: {s_te['gpt2'] / s_en['gpt2']:.1f}x\n")

    ratio = sum(te_totals) / sum(en_totals)
    print(f'Overall: {sum(te_totals)} Telugu vs {sum(en_totals)} English GPT-2 tokens '
          f'= {ratio:.1f}x inflation across {len(PAIRS)} sentence pairs')

    x = range(len(PAIRS))
    _, ax = plt.subplots(1, figsize=(8, 4))
    ax.bar([i - 0.2 for i in x], en_totals, width=0.4, label='English')
    ax.bar([i + 0.2 for i in x], te_totals, width=0.4, label='Telugu')
    ax.set(xticks=list(x), xticklabels=[f'Pair {i+1}' for i in x],
           ylabel='GPT-2 tokens', title=f'Same meaning, {ratio:.1f}x the tokens in Telugu')
    ax.legend()
    plt.tight_layout()
    plt.savefig('part2_telugu_cost.png')
    plt.close()
    print('saved: part2_telugu_cost.png')
    return ratio


if __name__ == '__main__':
    main()
