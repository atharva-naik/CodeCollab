import json
import nltk
from typing import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_corpus_bleu(path: str) -> Tuple[float, float]:
    with open(path, "r") as f:
        list_of_references = []
        list_of_hypotheses = []
        list_of_references_trunc = defaultdict(lambda:[])
        list_of_hypotheses_trunc = defaultdict(lambda:[])
        list_of_references_first_excluded = []
        list_of_hypotheses_first_excluded = []
        num_tokens_trunc = [10, 20, 50, 100]

        prev_id = None
        for line in tqdm(f):
            line = line.strip()
            rec = json.loads(line)
            id = rec['id']
            if prev_id == id:
                list_of_hypotheses_first_excluded.append(hypothesis)
                list_of_references_first_excluded.append(references)
            reference = rec["true"].split()
            hypothesis = rec["pred"].split()
            references = [reference]
            list_of_references.append(references)
            list_of_hypotheses.append(hypothesis)
            for num_tokens in num_tokens_trunc:
                list_of_references_trunc[num_tokens].append([reference[:num_tokens]])
                list_of_hypotheses_trunc[num_tokens].append(hypothesis[:num_tokens])
            prev_id = id
    
    total_bleu = nltk.translate.bleu_score.corpus_bleu(
        list_of_references, 
        list_of_hypotheses,
    )
    first_excluded_bleu = nltk.translate.bleu_score.corpus_bleu(
        list_of_references_first_excluded, 
        list_of_hypotheses_first_excluded,
    )
    trunc_bleu = {}
    for num_tokens in num_tokens_trunc:
        trunc_bleu[num_tokens] = nltk.translate.bleu_score.corpus_bleu(
            list_of_references_trunc[num_tokens], 
            list_of_hypotheses_trunc[num_tokens],
        )

    return total_bleu, first_excluded_bleu, trunc_bleu

def plot_truncated_bleus(trunc_bleu_scores: Dict[int, float]):
    plt.clf()
    plt.xlabel("Num tokens truncated at")
    plt.ylabel("BELU score")
    plt.xticks(
        list(trunc_bleu_scores.keys()),
        labels=list(trunc_bleu_scores.keys()),
    )
    plt.plot(list(trunc_bleu_scores.keys()),
             list(trunc_bleu_scores.values()),
             color="red")
    for i, key in enumerate(trunc_bleu_scores):
        value = trunc_bleu_scores[key]
        plt.annotate(f"{100*value:.2f}", (key, value))
    plt.title("Truncated input BLEU scores for InCoder markdown infilling")
    plt.tight_layout()
    plt.savefig("./plots/incoder_infill_markdown_truncated_bleu_scores.png")

if __name__ == "__main__":
    bleu_score, first_excluded_score, trunc_bleu_scores = compute_corpus_bleu("./analysis/incoder_markdown_infill_val.jsonl")
    print(f'''BLEU score: 
total: {bleu_score}
first excluded: {first_excluded_score}
truncated input scores: {trunc_bleu_scores}''')
    plot_truncated_bleus(trunc_bleu_scores)