import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance
import logging
import joblib
import os

class GaussianDistributionEstimator:
    def __init__(self, n_components=1, **kwargs):
        self.model = GaussianMixture(n_components=n_components, **kwargs)

    def fit(self, X):
        self.model.fit(X)

    def score_samples(self, X):
        return -self.model.score_samples(X) # 返回样本的负对数似然值
        
    def save_model(self, file_path):
        joblib.dump(self.model, file_path)
        
    def load_model(self, file_path):
        self.model = joblib.load(file_path)

class PseudoLabeler:
    def __init__(self, n_occ_models=5, n_components=1, **kwargs):
        self.occ_models = [GaussianDistributionEstimator(n_components=n_components, **kwargs) for _ in range(n_occ_models)]
        self.thresholds_pos = [None] * n_occ_models
        self.thresholds_neg = [None] * n_occ_models
        self.epoch_pseudo_labels = None  # 用于保存每个epoch的伪标签
        self.iteration_pseudo_labels = None  # 用于保存每次迭代的伪标签

    # def fit(self, X, model_dir='gs_models'):
    #     logging.info("开始拟合高斯混合模型...")
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
            
    #     # 直接使用未标记的数据来拟合各个高斯分布估计器
    #     for idx, model in enumerate(self.occ_models):
    #         model_path = os.path.join(model_dir, f'gmm_model_{idx}.joblib')
    #         if os.path.exists(model_path):
    #             logging.info(f"加载模型{idx+1}/{len(self.occ_models)}...")
    #             model.load_model(model_path)
    #         else:
    #             model.model.set_params(max_iter=200)  # 增加迭代次数以便收敛
    #             model.fit(X)
    #             model.save_model(model_path)
    #             logging.info(f"模型 {idx+1}/{len(self.occ_models)} 拟合成功。")
    #     logging.info("所有高斯混合模型拟合完成。")

    def fit(self, X):
        logging.info("开始拟合高斯混合模型...")
        # 直接使用未标记的数据来拟合各个高斯分布估计器
        for idx, model in enumerate(self.occ_models):
            model.model.set_params(max_iter=200)  # 增加迭代次数以便收敛
            model.fit(X)
            logging.info(f"模型 {idx+1}/{len(self.occ_models)} 拟合成功。")
        logging.info("所有高斯混合模型拟合完成。")

    def _partial_match(self, source, target):
        # 使用Wasserstein距离实现部分匹配
        min_dist = np.inf
        best_threshold = None
        for threshold in np.linspace(source.min(), source.max(), 100):
            dist = wasserstein_distance(source[source > threshold], target)
            if dist < min_dist:
                min_dist = dist
                best_threshold = threshold
        return best_threshold

    def update_iteration_pseudo_labels(self, features):
        # 在每次迭代后更新迭代伪标签
        self.iteration_pseudo_labels = self.generate_pseudo_labels(features)

    def update_epoch_pseudo_labels(self, features):
        # 在每个epoch开始时更新周期伪标签
        self.epoch_pseudo_labels = self.generate_pseudo_labels(features)

    def generate_pseudo_labels(self, features):
        # 确保特征数据从 CUDA 转移到 CPU，并转换为 NumPy 数组，只进行一次转换
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features  # 假设 features 已经是 NumPy 数组

        votes = np.zeros((len(features_np), len(self.occ_models)), dtype=int)
        for idx, model in enumerate(self.occ_models):
            scores = model.score_samples(features_np)
            # 假设阈值选择为模型分数的中位数
            threshold = np.median(scores)
            votes[:, idx] = scores > threshold

        # 确定伪标签: 多数模型同意的情况下赋予伪标签1，否则为0
        pseudo_labels = np.mean(votes, axis=1) > 0.5
        return pseudo_labels.astype(int)
