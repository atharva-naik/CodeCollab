# make t-SNE plot to compare regular code representation and obfuscated code representation
import json
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model.code_similarity_retrievers.dense import CodeBERTDenseSearcher

# main
if __name__ == "__main__":
    dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_CodeSearch2_CosSim/best_model.pt", 
        faiss_index_path="./dense_indices/codebert_cos_sim.index",
    )
    # vecs = dense_searcher.dense_index.reconstruct_n(0,10000)
    obf_dense_searcher = CodeBERTDenseSearcher(
        ckpt_path="./experiments/CoNaLa_CSN_CodeBERT_ObfCodeSearch4_CosSim/best_model.pt", 
        faiss_index_path="./dense_indices/codebert_obf_cos_sim.index",
    )
    # obf_vecs = dense_searcher.dense_index.reconstruct_n(0,10000)
    
    # sample a few 100 random codes.
    code_KB = json.load(open("./JuICe_train_code_KB.json"))
    codes = list(code_KB.keys())

    # sampled_codes = [codes[i] for i in random.sample(range(len(codes)), k=100)]
    sampled_codes = [codes[i] for i in random.sample(range(len(codes)), k=10)]

    vecs = dense_searcher.encode(sampled_codes, use_cos_sim=True)
    tsne_vecs = TSNE(n_components=2, learning_rate='auto',
                     init='random', perplexity=5).fit_transform(vecs)
    obf_vecs = obf_dense_searcher.encode(sampled_codes, use_cos_sim=True, obf_code=True)
    tsne_obf_vecs = TSNE(n_components=2, learning_rate='auto',
                         init='random', perplexity=5).fit_transform(obf_vecs)
    
    plt.clf()
    plt.scatter(tsne_vecs[:,0], tsne_vecs[:,1], label="regular")
    plt.scatter(tsne_obf_vecs[:,0], tsne_obf_vecs[:,1], label="obfuscated")
    plt.legend(loc="upper right")
    plt.savefig("./plots/effect_of_obfuscation_tsne.png")