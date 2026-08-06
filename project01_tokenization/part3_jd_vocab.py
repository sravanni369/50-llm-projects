"""Part 3: how do data-analyst skill terms fragment under GPT-2 BPE?

Snippets are representative requirement lines written for this analysis
(not scraped postings). The question: which resume/JD skill terms survive
as single tokens and which shatter into pieces?
"""

from schemes import gpt2_tokenizer, scheme_counts

JD_SNIPPETS = [
    'Proficiency in SQL, Python, and Tableau required for dashboard development.',
    'Experience with Excel pivot tables, Power BI, and statistical analysis.',
    'Build ETL pipelines and machine learning models using PyTorch and scikit-learn.',
]

SKILL_TERMS = ['SQL', 'Python', 'Tableau', 'Excel', 'PyTorch', 'scikit-learn',
               'ETL', 'Power BI', 'dashboard', 'statistics']


def fragmentation(term, tok):
    """Token pieces for a term as it appears mid-sentence (leading space)."""
    ids = tok.encode(' ' + term)
    return [tok.decode([i]) for i in ids]


def main():
    tok = gpt2_tokenizer()

    for snip in JD_SNIPPETS:
        counts = scheme_counts(snip)
        print(f'{snip}')
        print(f"    char {counts['char']['total']} | word {counts['word']['total']} "
              f"| gpt2 {counts['gpt2']['total']} tokens\n")

    print(f'{"term":<15} {"pieces":<8} tokens')
    print('-' * 50)
    for term in SKILL_TERMS:
        pieces = fragmentation(term, tok)
        print(f'{term:<15} {len(pieces):<8} {pieces}')


if __name__ == '__main__':
    main()
