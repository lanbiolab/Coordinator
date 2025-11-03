import argparse
import os
import numpy as np
import gc
import torch

from data_loader import load_dataset
from agents import FeatureViewAgent, CLIPSemanticAgent, BlipSemanticAgent, LlavaSemanticAgent, MambaSemanticAgent
from coordinator import SemanticAwareCoordinator
from utils import set_global_seeds

def run_experiment(dataset_name, data_path, sample_size, n_clusters, seed, semantic_model_name):
    """
    运行单次实验。
    """
    print(f"\n=== Starting experiment with seed {seed}, semantic_model: {semantic_model_name} ===")
    set_global_seeds(seed)
    
    # 1. 加载数据
    data_X, true_labels = load_dataset(dataset_name, data_path, sample_size)
    if n_clusters is None: 
        n_clusters = len(np.unique(true_labels))

    # 2. 初始化视图代理
    view_agents = [FeatureViewAgent(f"View-{i + 1}", X, n_clusters=n_clusters) for i, X in enumerate(data_X)]

    # 3. 根据名称选择并初始化语义代理
    if semantic_model_name.lower() == 'clip':
        semantic_agent = CLIPSemanticAgent(dataset_name, n_clusters=n_clusters)
    elif semantic_model_name.lower() == 'blip':
        semantic_agent = BlipSemanticAgent(dataset_name, n_clusters=n_clusters, model_name="Salesforce/blip-itm-base-coco")
    elif semantic_model_name.lower() == 'llava':
        semantic_agent = LlavaSemanticAgent(dataset_name, n_clusters=n_clusters)
    elif semantic_model_name.lower() == 'mamba':
        semantic_agent = MambaSemanticAgent(dataset_name, n_clusters=n_clusters)
    else:
        raise ValueError(f"未知的语义模型: {semantic_model_name}")

    print(f"--- Coordinator initialized with {semantic_agent.id} ---")
    
    # 4. 初始化并运行协调器
    coordinator = SemanticAwareCoordinator(view_agents, semantic_agent, n_clusters, dataset_name, seed)
    coordinator.integrate_views(max_iter=10, true_labels=true_labels)
    
    # 5. 评估并返回结果
    nmi_val, ari_val, acc_val, f1_val = coordinator.evaluate_performance(true_labels)

    # 清理内存
    del coordinator, view_agents, semantic_agent, data_X, true_labels
    gc.collect()
    torch.cuda.empty_cache()

    return nmi_val, ari_val, acc_val, f1_val


def main(args):
    """
    主函数，用于执行多次实验并计算平均结果。
    """
    results = {'nmi': [], 'ari': [], 'acc': [], 'f1': []}
    for seed in args.seeds:
        nmi_val, ari_val, acc_val, f1_val = run_experiment(
            args.dataset, args.data_path, args.sample_size, args.n_clusters, seed, args.semantic_model
        )
        results['nmi'].append(nmi_val)
        results['ari'].append(ari_val)
        results['acc'].append(acc_val)
        results['f1'].append(f1_val)

    print("\n" + "=" * 60)
    print(f"Final Results for {args.dataset} with {args.semantic_model} after {len(args.seeds)} runs")
    print("=" * 60)

    for metric in ['acc', 'nmi', 'ari', 'f1']:
        values = np.array(results[metric])
        if len(values) > 0 and not np.all(values == 0):
            print(f"{metric.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized Semantic-Aware Multi-view Clustering')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Scene15', 'LandUse21', 'Reuters', 'Hdigit', 'cub_googlenet', "RGB-D", "Cora", "ALOI",
                                 "Caltech101", "NUSWIDE", "CCV", "SUNRGBD", "cifar100", "YouTubeFace", "VGGFace"])
    parser.add_argument('--data_path', type=str, default='./data', help="存放数据集文件的路径")
    parser.add_argument('--semantic_model', type=str, default='clip', choices=['clip', 'blip', 'llava', 'mamba'],
                        help='选择用于语义引导的多模态或语言大模型 (clip, blip, llava, mamba)')
    parser.add_argument('--sample_size', type=int, default=None, help="对数据集进行子采样的大小")
    parser.add_argument('--n_clusters', type=int, default=None, help="聚类的簇数，如果为None则从数据标签中推断")
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help="用于实验的随机种子列表，可进行多次实验取平均值")
    args = parser.parse_args()

    # 确保数据路径存在
    if not os.path.exists(args.data_path):
        os.makedirs(args.data_path)
        print(f"创建数据目录: {args.data_path}")
        print("请确保您的数据集文件 (例如 scene15.mat) 存放在此目录中。")

    main(args)