import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
from scipy.optimize import linear_sum_assignment
from scipy import sparse
from scipy.sparse import csgraph

def set_global_seeds(seed=0):
    """设置全局随机种子以确保实验可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def cluster_acc(y_true, y_pred):
    """计算聚类准确度 (ACC)。"""
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size

def visualize_clustering(embeddings, true_labels, title, filename, method='umap', n_clusters=None):
    """使用UMAP对嵌入进行降维并可视化聚类结果。"""
    plt.figure(figsize=(10, 8))
    is_similarity = embeddings.shape[0] == embeddings.shape[1] and np.allclose(embeddings, embeddings.T,
                                                                               atol=1e-6) if not sparse.issparse(
        embeddings) else embeddings.shape[0] == embeddings.shape[1]

    if is_similarity:
        print(f"Performing spectral embedding for similarity matrix of size {embeddings.shape}")
        try:
            if sparse.issparse(embeddings):
                embeddings = embeddings.toarray()
            laplacian = csgraph.laplacian(embeddings, normed=True)
            _, eigenvectors = np.linalg.eigh(laplacian)
            if n_clusters is None:
                n_clusters = len(np.unique(true_labels))
            spectral_emb = eigenvectors[:, -n_clusters:]
            reducer = umap.UMAP(random_state=42)
            embeddings_2d = reducer.fit_transform(spectral_emb)
        except Exception as e:
            print(f"Spectral embedding failed: {e}")
            reducer = umap.UMAP(metric='precomputed', random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
    else:
        reducer = umap.UMAP(random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5']
    unique_labels = np.unique(true_labels)
    for i, label in enumerate(unique_labels):
        idx = true_labels == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1],
                    color=colors[i % len(colors)],
                    label=f'Class {label}', alpha=0.7, s=30)
    plt.title(title, fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best', fontsize=10)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {filename}")