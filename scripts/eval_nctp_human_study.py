import json
import numpy as np
import pandas as pd
from typing import *
import matplotlib.pyplot as plt
from collections import defaultdict

human_annot = pd.read_excel("./analysis/nctp_human_study_annot_val.xlsx")
true_labels = json.load(open("./analysis/nctp_human_study_val_trues.json"))

def compute_acc(human_annot: pd.DataFrame, true_labels: Dict[str, Tuple[str, str]]):
    ids =  human_annot.id
    preds = list(human_annot.pred)
    tot, matches = 0, 0
    ctxt_sizes = list(human_annot.context_size)
    tot_context_size = defaultdict(lambda:0)
    matches_context_size = defaultdict(lambda:0)
    conf_mat = defaultdict(lambda:defaultdict(lambda:0))
    
    for i, id in enumerate(ids):
        pred = preds[i]
        if isinstance(pred, float): continue
        pred = pred.strip()
        if pred == "skip": continue
        true = true_labels[str(id)][0].strip()
        tot += 1
        ctxt_size = ctxt_sizes[i]
        matches += int(pred == true)
        tot_context_size[ctxt_size] += 1
        matches_context_size[ctxt_size] += int(pred == true)
        conf_mat[true][pred] += 1
    tot_context_size = dict(tot_context_size)
    matches_context_size = dict(matches_context_size)
    for k in tot_context_size:
        matches_context_size[k] = matches_context_size[k]/tot_context_size[k]

    return (100*matches)/tot, matches, matches_context_size, dict(conf_mat)

def plot_acc_by_ctxt_size(accs: List[float],
                          ctxt_sizes: List[int]=[2,5,10]):
    plt.clf()
    x = [1,2,3]
    plt.bar(x, accs, width=0.5)
    plt.xticks(x, labels=ctxt_sizes)
    plt.xlabel("context size")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs context size")
    plt.savefig("./plots/ctxt_size_accs.png")

def plot_conf_mat(conf_mat):
    plt.clf()
    x = ["markdown", "code"]
    y = ["markdown", "code"]
    z = np.zeros((2,2))
    fig, ax = plt.subplots(figsize=(3,3))
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    for i in range(2):
        for j in range(2): 
            k1 = x[i]
            k2 = y[j]
            z[i][j] = conf_mat[k1][k2]
    ax.set_xticks([0,1], labels=x)
    ax.set_yticks([0,1], labels=y)
    ax.matshow(z, cmap="Blues")
    for i in range(2):
        for j in range(2): 
            k1 = x[i]
            k2 = y[j]
            ax.text(j, i, '{:0g}'.format(z[i][j]))
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("./plots/conf_mat.png")

def plot_conf_mat_from_seq(data: List[dict]):
    from collections import defaultdict

    conf_mat = np.zeros((2,3))
    x = ["markdown", "code"]
    y = ["markdown", "code", "ppe"]
    for rec in data:
        p = rec['pred']
        t = rec['true']
        if p not in ["code", "markdown"]:
            p = "ppe"
        i = x.index(t)
        j = y.index(p)
        conf_mat[i][j] += 1
    plt.clf()
    fig, ax = plt.subplots(figsize=(4,6))
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_xticks([0,1], labels=x)
    ax.set_yticks([0,1,2], labels=y)
    ax.matshow(conf_mat, cmap="Blues")
    for i in range(2):
        for j in range(3): 
            k1 = x[i]
            k2 = y[j]
            ax.text(j, i, '{:0g}'.format(conf_mat[i][j]))
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("./plots/conf_mat_ntcp.png")