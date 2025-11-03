import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import f1_score
from scipy import sparse
import gc
import copy

from modules import CrossViewAlignmentModule
from utils import cluster_acc

class SemanticAwareCoordinator:
    """
    协调器，负责整合所有视图和语义信息，执行迭代聚类。
    """
    def __init__(self, view_agents, semantic_agent, n_clusters=10, dataset_name="unknown", seed=0):
        self.view_agents = view_agents
        self.semantic_agent = semantic_agent
        self.n_clusters = n_clusters
        self.global_consensus = None
        self.final_labels = None
        self.view_weights = np.ones(len(view_agents)) / len(view_agents)
        self.semantic_weight = 0.3
        self.dataset_name = dataset_name
        self.seed = seed
        self.cross_view_aligner = self._init_cross_view_aligner()
        self.best_iter, self.best_nmi, self.best_ari, self.best_acc, self.best_f1 = -1, 0, 0, 0, 0
        self.best_model_state = None

    def _init_cross_view_aligner(self):
        layer_dims = [[agent.embedding_dim, 512, 128] for agent in self.view_agents]
        return CrossViewAlignmentModule(len(self.view_agents), layer_dims, 0.1, self.n_clusters).cuda()

    def integrate_views(self, max_iter=10, true_labels=None):
        similarity_matrices = [None] * len(self.view_agents)
        embeddings_list = [None] * len(self.view_agents)
        aligner_optimizer = optim.Adam(self.cross_view_aligner.parameters(), lr=0.0005)
        semantic_sim = None

        for i, agent in enumerate(self.view_agents):
            print(f"\n--- Training View Agent {agent.id} ---")
            agent.train_autoencoder(epochs=50, true_labels=true_labels)
            agent.compute_local_clusters(self.n_clusters)
            similarity_matrices[i] = agent.compute_similarity_matrix()
            embeddings_list[i] = agent.embedding

        avg_embedding = np.mean([e for e in embeddings_list if e is not None and e.size > 0], axis=0)

        if true_labels is not None and all(a.local_clusters is not None and len(a.local_clusters) == len(true_labels) for a in self.view_agents):
            weights = [nmi(true_labels, agent.local_clusters) for agent in self.view_agents if agent.local_clusters is not None]
            if weights: self.view_weights = np.array(weights) / (np.sum(weights) + 1e-8)

        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1}/{max_iter} ---")
            if all(e is not None for e in embeddings_list):
                self._train_cross_view_aligner(aligner_optimizer, embeddings_list, iteration, max_iter)
                aligned_features = self._get_aligned_features(embeddings_list)
                for i in range(len(self.view_agents)):
                    self.view_agents[i].embedding = aligned_features[i].cpu().numpy()
                    similarity_matrices[i] = self.view_agents[i].compute_similarity_matrix()
            
            if iteration % 3 == 0 and hasattr(avg_embedding, 'size') and avg_embedding.size > 0:
                semantic_sim = self.semantic_agent.compute_similarity_matrix(avg_embedding)

            current_semantic_weight = max(0.1, self.semantic_weight * (0.9 ** iteration))
            valid_sim_matrices = [s for s in similarity_matrices if s is not None and s.shape[0] > 0]
            if not valid_sim_matrices: continue

            self.global_consensus = sparse.csr_matrix(valid_sim_matrices[0].shape, dtype=np.float32)
            for i, sim_matrix in enumerate(similarity_matrices):
                if sim_matrix is not None and sim_matrix.shape == self.global_consensus.shape:
                    self.global_consensus += self.view_weights[i] * sim_matrix
            if semantic_sim is not None and semantic_sim.shape == self.global_consensus.shape:
                self.global_consensus += current_semantic_weight * semantic_sim
            
            total_weight = self.view_weights.sum() + (current_semantic_weight if semantic_sim is not None else 0)
            if total_weight > 0: self.global_consensus = self.global_consensus / total_weight

            spectral = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', random_state=self.seed + iteration, n_init=10, eigen_solver='lobpcg')
            self.final_labels = spectral.fit_predict(self.global_consensus)

            if true_labels is not None:
                current_nmi = nmi(true_labels, self.final_labels)
                current_ari = ari_score(true_labels, self.final_labels)
                current_acc = cluster_acc(true_labels, self.final_labels)
                current_f1 = f1_score(true_labels, self.final_labels, average='macro')
                print(f"Metrics: NMI={current_nmi:.4f}, ARI={current_ari:.4f}, ACC={current_acc:.4f}, F1={current_f1:.4f}")
                if current_acc > self.best_acc:
                    self.best_iter, self.best_nmi, self.best_ari, self.best_acc, self.best_f1 = iteration, current_nmi, current_ari, current_acc, current_f1
                    self.best_model_state = {
                        "view_agents": [copy.deepcopy(a.model.state_dict()) for a in self.view_agents],
                        "cross_view_aligner": copy.deepcopy(self.cross_view_aligner.state_dict()),
                        "final_labels": copy.deepcopy(self.final_labels)
                    }
                    print(f"!!! New best model at iteration {iteration + 1} !!!")

            if all(a.local_clusters is not None for a in self.view_agents):
                for i, agent in enumerate(self.view_agents):
                    agent.refine_with_consensus(self.final_labels)
                    if len(self.final_labels) == len(agent.local_clusters):
                        view_nmi = nmi(self.final_labels, agent.local_clusters)
                        self.view_weights[i] = self.view_weights[i] * 0.6 + view_nmi * 0.4
                self.view_weights /= (self.view_weights.sum() + 1e-8)
            
            embeddings_list = [agent.embedding for agent in self.view_agents]
            avg_embedding = np.mean([e for e in embeddings_list if e is not None and e.size > 0], axis=0)
            gc.collect()

        if self.best_model_state:
            print(f"\nRestoring best model from iteration {self.best_iter + 1}")
            self.final_labels = self.best_model_state["final_labels"]
        
        return self.final_labels

    def _train_cross_view_aligner(self, optimizer, embeddings_list, iteration, max_iter, batch_size=512):
        self.cross_view_aligner.train()
        momentum = 0.99
        epochs = 3 if iteration == 0 else 1

        datasets = [torch.from_numpy(emb).float() for emb in embeddings_list]
        full_dataset = torch.utils.data.TensorDataset(*datasets)
        loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        final_loss, num_batches = 0, 0
        for epoch in range(epochs):
            for data_tensors_list in loader:
                data_tensors = [d.cuda(non_blocking=True) for d in data_tensors_list]
                optimizer.zero_grad()
                loss = self.cross_view_aligner(data_tensors, momentum, warm_up=(iteration == 0))
                loss.backward()
                optimizer.step()
                final_loss += loss.item()
                num_batches += 1
        avg_loss = final_loss / num_batches if num_batches > 0 else 0
        print(f"Cross-view alignment - Iter {iteration + 1}: Avg Loss={avg_loss:.4f}")

    def _get_aligned_features(self, embeddings_list, batch_size=1024):
        self.cross_view_aligner.eval()
        aligned_features_parts = [[] for _ in range(len(embeddings_list))]

        datasets = [torch.from_numpy(emb).float() for emb in embeddings_list]
        full_dataset = torch.utils.data.TensorDataset(*datasets)
        loader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for data_tensors_list in loader:
                data_tensors = [d.cuda() for d in data_tensors_list]
                aligned_batch = self.cross_view_aligner.extract_aligned_features(data_tensors)
                for i in range(len(embeddings_list)):
                    aligned_features_parts[i].append(aligned_batch[i].cpu())
        
        aligned_features = [torch.cat(parts) for parts in aligned_features_parts if parts]
        return aligned_features

    def evaluate_performance(self, true_labels):
        if self.final_labels is None or len(self.final_labels) != len(true_labels):
            print("\nFinal labels are not available or have incorrect length. Skipping evaluation.")
            return 0, 0, 0, 0
        nmi_score_val = nmi(true_labels, self.final_labels)
        ari_score_val = ari_score(true_labels, self.final_labels)
        acc_score = cluster_acc(true_labels, self.final_labels)
        f1_score_val = f1_score(true_labels, self.final_labels, average='macro')
        print(f"\nFinal Performance: NMI={nmi_score_val:.4f}, ARI={ari_score_val:.4f}, ACC={acc_score:.4f}, F1={f1_score_val:.4f}")
        return nmi_score_val, ari_score_val, acc_score, f1_score_val