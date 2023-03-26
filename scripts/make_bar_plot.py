from typing import *
def make_bar_plot(k_cts: Dict[str, int]):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title(f"Topic topics in NotebookCDG preds")
    x = range(1, len(k_cts)+1)
    y = list(k_cts.values())
    plt.bar(x, y, color="orange")
    plt.ylabel("freq")
    plt.xlabel("topics")
    plt.xticks(x, labels=list(k_cts.keys()), rotation=90)
    plt.tight_layout()
    plt.savefig(f"plots/HAConvGNN_pred_topics.png")