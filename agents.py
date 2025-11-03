import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from scipy import sparse
from scipy.sparse import csgraph
import gc
from abc import ABC, abstractmethod
import warnings

# 导入多模态模型库
import clip
from transformers import BlipProcessor, BlipForImageTextRetrieval, logging as hf_logging
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, MambaForCausalLM

from modules import ContrastiveViewEncoder, InstanceLoss
from utils import cluster_acc

# 减少不必要的transformers库的日志输出
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class FeatureViewAgent:
    """处理单个数据视图的代理。"""
    def __init__(self, view_id, view_data, n_clusters=10, embedding_dim=128):
        self.id = view_id
        self.data = view_data
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = view_data.shape[1]
        self.model = ContrastiveViewEncoder(input_dim, embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        self.contrastive_criterion = InstanceLoss(batch_size=256, temperature=0.1, device=self.device)
        self.local_clusters = None
        self.similarity_matrix = None
        self.credibility = 1.0
        self.embedding = None

    def train_autoencoder(self, epochs=50, batch_size=256, true_labels=None):
        self.model.train()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.data).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

        for epoch in range(epochs):
            total_loss = 0.0
            for (batch_data,) in loader:
                inputs = batch_data.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                _, proj1 = self.model(inputs)
                _, proj2 = self.model(inputs)
                loss = self.contrastive_criterion(proj1, proj2)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0

            if true_labels is not None and (epoch + 1) % 10 == 0:
                self.embedding = self.get_embedding_numpy(batch_size)
                current_clusters = KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42).fit_predict(self.embedding)
                current_nmi = nmi(true_labels, current_clusters)
                current_acc = cluster_acc(true_labels, current_clusters)
                print(f"View {self.id} - Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, NMI: {current_nmi:.4f}, ACC: {current_acc:.4f}")
        
        self.embedding = self.get_embedding_numpy(batch_size)
        gc.collect()

    def get_embedding_numpy(self, batch_size=1024):
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(self.data), batch_size):
                batch = torch.from_numpy(self.data[i:i + batch_size]).float().to(self.device)
                emb, _ = self.model(batch)
                embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.array([])

    def compute_local_clusters(self, n_clusters=None):
        if n_clusters is None: n_clusters = self.n_clusters
        if self.embedding is None: self.train_autoencoder()
        if self.embedding.shape[0] == 0: return np.array([])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.local_clusters = kmeans.fit_predict(self.embedding)
        return self.local_clusters

    def compute_similarity_matrix(self, batch_size=1024, top_k=50):
        if self.embedding is None: self.train_autoencoder()
        n_samples = self.embedding.shape[0]
        if n_samples == 0: return sparse.csr_matrix((0, 0))
        self.similarity_matrix = sparse.lil_matrix((n_samples, n_samples), dtype=np.float32)
        norm_embedding = self.embedding / (np.linalg.norm(self.embedding, axis=1, keepdims=True) + 1e-8)

        for i in range(0, n_samples, batch_size):
            batch_emb = norm_embedding[i:i + batch_size]
            sim_batch = batch_emb @ norm_embedding.T
            k = min(top_k, sim_batch.shape[1])
            top_k_indices = np.argpartition(sim_batch, -k, axis=1)[:, -k:]
            rows = np.arange(i, i + batch_emb.shape[0]).repeat(k)
            cols = top_k_indices.flatten()
            vals = sim_batch[np.arange(batch_emb.shape[0])[:, None], top_k_indices].flatten()
            valid_mask = vals > 0.5
            self.similarity_matrix[rows[valid_mask], cols[valid_mask]] = vals[valid_mask]

        self.similarity_matrix = self.similarity_matrix.tocsr()
        self.similarity_matrix = (self.similarity_matrix + self.similarity_matrix.T) / 2
        try:
            laplacian = csgraph.laplacian(self.similarity_matrix, normed=True)
            self.similarity_matrix = sparse.identity(n_samples) - laplacian
        except Exception: pass
        gc.collect()
        return self.similarity_matrix

    def refine_with_consensus(self, consensus_labels, batch_size=256):
        self.fine_tune_autoencoder(consensus_labels, epochs=5, batch_size=batch_size)
        self.embedding = self.get_embedding_numpy(batch_size)
        self.compute_similarity_matrix()
        self.compute_local_clusters(self.n_clusters)

    def fine_tune_autoencoder(self, pseudo_labels, epochs=5, batch_size=256):
        self.model.train()
        pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(self.data).float(), pseudo_labels_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        class FineTuneModel(nn.Module):
            def __init__(self, encoder, embedding_dim, n_clusters):
                super(FineTuneModel, self).__init__()
                self.encoder = encoder
                self.clustering_layer = nn.Linear(embedding_dim, n_clusters)
            def forward(self, x):
                features, _ = self.encoder(x)
                return self.clustering_layer(features)

        fine_tune_model = FineTuneModel(self.model, self.embedding_dim, self.n_clusters).to(self.device)
        optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.0001, weight_decay=1e-5)
        clustering_criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                cluster_output = fine_tune_model(inputs)
                loss = clustering_criterion(cluster_output, labels)
                loss.backward()
                optimizer.step()
        self.model = fine_tune_model.encoder
        gc.collect()

class BaseSemanticAgent(ABC):
    """语义代理的抽象基类。"""
    def __init__(self, dataset_name, n_clusters, class_names=None):
        self.id = "Base"
        self.dataset_name = dataset_name
        self.n_clusters = n_clusters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.similarity_matrix = None
        self.semantic_embeddings = None
        self.projection_matrix = None

        raw_class_names = self._get_class_names(class_names)
        if len(raw_class_names) != self.n_clusters:
            warnings.warn(f"[{self.id}] 类别数量({self.n_clusters})与提供的类别名数量({len(raw_class_names)})不匹配。将使用通用类别名。")
            self.class_names = [f"class_{i}" for i in range(self.n_clusters)]
        else:
            self.class_names = raw_class_names
        self._generate_semantic_embeddings()

    @abstractmethod
    def _generate_semantic_embeddings(self):
        pass

    def _align_to_semantic_space(self, embeddings, batch_size=1024):
        feature_dim = embeddings.shape[1]
        semantic_dim = self.semantic_embeddings.shape[1]
        if feature_dim != semantic_dim:
            if self.projection_matrix is None or self.projection_matrix.shape != (feature_dim, semantic_dim):
                print(f"[{self.id}] 创建从 {feature_dim}D 到 {semantic_dim}D 的投影矩阵。")
                np.random.seed(42)
                self.projection_matrix = np.random.randn(feature_dim, semantic_dim).astype(np.float32) * 0.01
                self.projection_matrix /= (np.linalg.norm(self.projection_matrix, axis=0, keepdims=True) + 1e-8)
        
        n_samples = embeddings.shape[0]
        class_similarity = np.zeros((n_samples, self.n_clusters))
        norm_sem = self.semantic_embeddings / (np.linalg.norm(self.semantic_embeddings, axis=1, keepdims=True) + 1e-8)

        for i in range(0, n_samples, batch_size):
            batch_emb = embeddings[i:i + batch_size]
            if feature_dim != semantic_dim:
                batch_emb = batch_emb @ self.projection_matrix
            norm_emb_batch = batch_emb / (np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-8)
            full_sim = norm_emb_batch @ norm_sem.T
            num_cols_to_assign = min(full_sim.shape[1], self.n_clusters)
            class_similarity[i:i + batch_size, :num_cols_to_assign] = full_sim[:, :num_cols_to_assign]
        return class_similarity

    def compute_similarity_matrix(self, embeddings, batch_size=1024, top_k=50):
        print(f"[{self.id}] 正在构建语义引导矩阵...")
        class_similarity = self._align_to_semantic_space(embeddings, batch_size)
        n_samples = embeddings.shape[0]
        semantic_sim_sparse = sparse.lil_matrix((n_samples, n_samples), dtype=np.float32)

        for i in range(0, n_samples, batch_size):
            batch_class_sim = class_similarity[i:i + batch_size]
            sim_batch = batch_class_sim @ class_similarity.T
            k = min(top_k, sim_batch.shape[1])
            if k == 0: continue
            top_k_indices = np.argpartition(sim_batch, -k, axis=1)[:, -k:]
            rows = np.arange(i, i + batch_class_sim.shape[0]).repeat(k)
            cols = top_k_indices.flatten()
            vals = sim_batch[np.arange(batch_class_sim.shape[0])[:, None], top_k_indices].flatten()
            vals = np.exp(vals / 0.1)
            semantic_sim_sparse[rows, cols] = vals

        semantic_sim_sparse = semantic_sim_sparse.tocsr()
        row_sums = np.array(semantic_sim_sparse.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        inv_row_sums = sparse.diags(1.0 / row_sums)
        self.similarity_matrix = inv_row_sums @ semantic_sim_sparse
        gc.collect()
        return self.similarity_matrix
        
    def _get_class_names(self, class_names):
        if class_names is not None: return class_names
        class_map = {
            'Scene15': ['bedroom', 'suburb', 'industrial', 'kitchen', 'livingroom', 'coast', 'forest', 'highway', 'insidecity', 'mountain', 'office', 'opencountry', 'store', 'street', 'tallbuilding'],
            'LandUse21': ['agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt'],
            'Reuters': ['corporate', 'economics', 'government', 'markets', 'shipping', 'commodities'], "Hdigit": [str(i) for i in range(10)],
            'cub_googlenet': ['sparrow', 'robin', 'eagle', 'owl', 'penguin', 'parrot', 'flamingo', 'swan', 'woodpecker', 'hummingbird'],
            'Caltech101': ['accordion', 'airplanes', 'anchor', 'ant', 'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium', 'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'leopards', 'llama', 'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone', 'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang'],
            'cifar100': ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
            'YouTubeFace': ['albert_einstein', 'ayrton_senna', 'barack_obama', 'bill_clinton', 'bill_gates', 'claire_danes', 'cobe_bryant', 'cristiano_ronaldo', 'dalai_lama', 'david_beckham', 'george_w_bush', 'gerhard_schroeder', 'harrison_ford', 'hillary_clinton', 'jack_nicholson', 'jennifer_lopez', 'john_mccain', 'johnny_depp', 'julia_roberts', 'keanu_reeves', 'michael_jordan', 'mikhail_gorbachev', 'paris_hilton', 'pope_john_paul_ii', 'ronaldinho', 'saddam_hussein', 'sylvester_stallone', 'tiger_woods', 'tom_cruise', 'vladimir_putin', 'will_smith']
        }
        return class_map.get(self.dataset_name, [f"class_{i}" for i in range(self.n_clusters)])

class CLIPSemanticAgent(BaseSemanticAgent):
    """使用CLIP模型作为语义代理。"""
    def __init__(self, dataset_name, n_clusters, class_names=None, model_name="ViT-B/32"):
        self.model_name = model_name
        self.model, _ = clip.load(self.model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(dataset_name, n_clusters, class_names)
        self.id = f"CLIP-{model_name.replace('/', '-')}"

    def _generate_semantic_embeddings(self):
        print(f"[{self.id}] 正在生成文本嵌入...")
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.class_names]).to(self.device)
            text_features = self.model.encode_text(text_inputs)
        self.semantic_embeddings = text_features.cpu().numpy()

class BlipSemanticAgent(BaseSemanticAgent):
    """使用BLIP模型作为语义代理。"""
    def __init__(self, dataset_name, n_clusters, class_names=None, model_name="Salesforce/blip-itm-base-coco"):
        self.model_name = model_name
        print(f"[BLIP] 正在加载模型: {self.model_name}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForImageTextRetrieval.from_pretrained(self.model_name).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        super().__init__(dataset_name, n_clusters, class_names)
        self.id = f"BLIP-{model_name.split('/')[-1]}"

    def _generate_semantic_embeddings(self):
        print(f"[{self.id}] 正在生成文本嵌入...")
        with torch.no_grad():
            text_prompts = [f"a photo of a {c}" for c in self.class_names]
            inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_outputs = self.model.text_encoder(**inputs)
            text_features = text_outputs.last_hidden_state[:, 0, :]
        self.semantic_embeddings = text_features.cpu().numpy()

class LlavaSemanticAgent(BaseSemanticAgent):
    """使用LLaVA模型作为语义代理。"""
    def __init__(self, dataset_name, n_clusters, class_names=None, model_name="llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        print(f"[LLaVA] 正在加载模型: {self.model_name}...")
        self.model = LlavaForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model.eval()
        super().__init__(dataset_name, n_clusters, class_names)
        self.id = f"LLaVA-{model_name.split('/')[-1]}"

    def _generate_semantic_embeddings(self):
        print(f"[{self.id}] 正在生成文本嵌入...")
        all_features = []
        with torch.no_grad():
            for c in self.class_names:
                prompt = f"USER: Describe what a '{c}' is.\nASSISTANT:"
                inputs = self.processor(text=prompt, return_tensors="pt").to(self.device, torch.float16)
                outputs = self.model.language_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze()
                all_features.append(embedding.cpu().numpy())
        self.semantic_embeddings = np.array(all_features)

class MambaSemanticAgent(BaseSemanticAgent):
    """使用Mamba语言模型作为语义代理。"""
    def __init__(self, dataset_name, n_clusters, class_names=None, model_name="state-spaces/mamba-130m-hf"):
        self.model_name = model_name
        print(f"[Mamba] 正在加载模型: {self.model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("eleutherai/gpt-neox-20b")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = MambaForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        self.model.eval()
        super().__init__(dataset_name, n_clusters, class_names)
        self.id = f"Mamba-{model_name.split('/')[-1]}"

    def _generate_semantic_embeddings(self):
        print(f"[{self.id}] 正在生成文本嵌入...")
        all_features = []
        with torch.no_grad():
            for c in self.class_names:
                prompt = f"A photo of a {c}"
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs["input_ids"].to(self.device)
                outputs = self.model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embedding = hidden_states.mean(dim=1).squeeze()
                all_features.append(embedding.cpu().to(torch.float32).numpy())
        self.semantic_embeddings = np.array(all_features)