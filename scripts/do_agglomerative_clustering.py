# do Agglomerative Clustering of the seed plan operator names.
import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

# plot a dendogram from the fitted aglomerative clustering model: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, seed_plan_ops, **kwargs):
    """dendogram plotting code for agglomerative clustering."""
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    child_sets = [set() for i in range(len(counts))]
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        current_set = set()
        for child_idx in merge:
            if child_idx < n_samples:
                if seed_plan_ops[child_idx.item()] not in current_set:
                    current_count += 1  # leaf node
                    # exit()
                    current_set.add(seed_plan_ops[child_idx.item()])
            else:
                current_set = current_set.union(child_sets[child_idx - n_samples])
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
        child_sets[i] = current_set
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # print(model.children_[0])
    # print(model.distances_.shape)
    # print(linkage_matrix.shape)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    for i in range(len(child_sets)):
        child_sets[i] = sorted(list(child_sets[i]))
        assert counts[i] == len(child_sets[i]), f"counts[{i}] = {counts[i]} | len(child_sets[{i}]) = {len(child_sets[i])}"
    # print(child_sets)
    return child_sets

# main
if __name__ == "__main__":
    # load the seed plan operators:
    model_path = "CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim"
    seed_plan_ops = [str(rec["human"]) if str(rec["human"]).strip() != "nan" else str(rec["orig"]) for rec in pd.read_csv("./data/juice-dataset/seed_query_relabels.csv").to_dict("records") if str(rec["human"]) != "SKIP"]
    # print(seed_plan_ops)
    # exit()

    # encode seed queries
    dense_searcher = CodeBERTDenseSearcher(ckpt_path=f"./experiments/{model_path}/best_model.pt")
    vecs = dense_searcher.encode(seed_plan_ops, text_query=True)
    
    # do agglomerative clustering
    clusterer = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    clusterer = clusterer.fit(vecs)

    plt.title("Plan Operators Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    p = 3
    truncate_mode = "level"
    child_sets = plot_dendrogram(clusterer, seed_plan_ops, truncate_mode=truncate_mode, p=p)
    with open(f"./experiments/{model_path}/seed_plan_op_leaf_nodes_trunc-{truncate_mode}_p-{p}.json", "w") as f:
        json.dump(child_sets, f, indent=4)    
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    os.makedirs(os.path.join("plots", model_path), exist_ok=True)
    plot_save_path = os.path.join("plots", model_path, f"seed_plan_op_dendogram_trunc-{truncate_mode}_p-{p}.png")
    plt.savefig(plot_save_path)