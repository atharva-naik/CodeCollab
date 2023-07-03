# make t-SNE plot to compare regular code representation and obfuscated code representation
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import splrep, BSpline
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

# main
if __name__ == "__main__":
    random.seed(2023)
    dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt", 
        faiss_index_path="./dense_indices/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/cos_sim.index",
    )
    # vecs = dense_searcher.dense_index.reconstruct_n(0,10000)
    obf_dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/best_model.pt", 
        faiss_index_path="./dense_indices/codebert_obf_cos_sim.index",
    )
    # obf_vecs = dense_searcher.dense_index.reconstruct_n(0,10000)
    
    # sample `sampling_k` random codes.
    sampling_k: int = 25
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())

    # sampled_codes = [codes[i] for i in random.sample(range(len(codes)), k=100)]
    sampled_codes = [codes[i] for i in random.sample(range(len(codes)), k=sampling_k)]

    vecs: np.ndarray = dense_searcher.encode(sampled_codes, use_cos_sim=True)
    obf_vecs: np.ndarray = obf_dense_searcher.encode(sampled_codes, use_cos_sim=True, obf_code=True)
    # print(type(vecs))
    # print(vecs.shape)
    # print(len(vecs))
    all_tsne_vecs = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=5).fit_transform(np.concatenate([vecs, obf_vecs]))
    tsne_vecs = all_tsne_vecs[:sampling_k,:]
    tsne_obf_vecs = all_tsne_vecs[sampling_k:,:]
    assert len(tsne_vecs) == len(tsne_obf_vecs)
    # static matplotlib plot.

    # plot showing shift of points after obfuscation.
    plt.clf()
    plt.scatter(tsne_vecs[:,0], tsne_vecs[:,1], label="regular")
    plt.scatter(tsne_obf_vecs[:,0], tsne_obf_vecs[:,1], label="obfuscated")
    barrier_points = []
    for i in range(len(tsne_vecs)):
        plt.plot(
            [tsne_vecs[i][0], tsne_obf_vecs[i][0]],
            [tsne_vecs[i][1], tsne_obf_vecs[i][1]],
            linestyle="dashed", color="black",
        )
        x_Gi = (tsne_vecs[i][0] + tsne_obf_vecs[i][0])/2
        y_Gi = (tsne_vecs[i][1] + tsne_obf_vecs[i][1])/2
        barrier_points.append((x_Gi, y_Gi))
    barrier_points = sorted(barrier_points, reverse=False, key=lambda x: x[0])
    plt.legend(loc="upper right")
    plt.savefig("./plots/effect_of_obfuscation_tsne_shift.png")

    # plot showing barrier/boundary between obfuscated and regular code bert representation points.
    plt.clf()
    
    x_barrier = [i[0] for i in barrier_points]
    y_barrier = [i[1] for i in barrier_points]

    plt.scatter(tsne_vecs[:,0], tsne_vecs[:,1], label="regular")
    plt.scatter(tsne_obf_vecs[:,0], tsne_obf_vecs[:,1], label="obfuscated")
    plt.legend(loc="upper right")
    plt.plot(x_barrier, y_barrier, color="black")
    plt.savefig("./plots/effect_of_obfuscation_tsne_barrier.png")

    # plot showing shift of points after obfuscation and barrier/boundary between obfuscated and regular code bert representation points.
    plt.clf()
    plt.scatter(tsne_vecs[:,0], tsne_vecs[:,1], label="regular")
    plt.scatter(tsne_obf_vecs[:,0], tsne_obf_vecs[:,1], label="obfuscated")
    plt.legend(loc="upper right")
    for i in range(len(tsne_vecs)):
        plt.plot(
            [tsne_vecs[i][0], tsne_obf_vecs[i][0]],
            [tsne_vecs[i][1], tsne_obf_vecs[i][1]],
            linestyle="dashed", color="black",
        )
        x_Gi = (tsne_vecs[i][0] + tsne_obf_vecs[i][0])/2
        y_Gi = (tsne_vecs[i][1] + tsne_obf_vecs[i][1])/2
    plt.plot(x_barrier, y_barrier, color="black")
    plt.savefig("./plots/effect_of_obfuscation_tsne_shift_and_barrier.png")

    # make an interactive plot to show effect of obfuscation.
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]],
                        subplot_titles=('Effect of obfuscation'))
    # hoverable interactive scatterplot that displays the code when you hover over it.
    fig.add_trace(
        go.Scatter2(
            x=tsne_vecs[:,0],
            y=tsne_vecs[:,1],
            text=sampled_codes, 
            hovertemplate='%{text}'
        ), 
        row=1, 
        col=1
    )
    # fig.update_layout(title='Add Custom Data')
    fig.show()