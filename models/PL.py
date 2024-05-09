import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wasserstein_distance
from skimage.filters import threshold_otsu

from models.density import GaussianDensitySklearn, GaussianDensityTorch
from sklearn.model_selection import KFold


class PseudoLabeler:
    def __init__(self, num_PseudoLabeler, feature_extractor):
        self.features_list_0 = []
        self.features_list_1 = []
        self.X_train0 = None
        self.X_train1 = None
        self.models = []
        self.eta_p = []
        self.eta_n = []
        self.num_PseudoLabeler = num_PseudoLabeler
        self.feature_extractor = feature_extractor

    def _extract_features(self, inputs, targets, device):
        targets_indices0 = torch.where(targets == 0)[0]
        targets_indices1 = torch.where(targets == 1)[0]
        for targets_indices, features_list in zip([targets_indices0, targets_indices1], [self.features_list_0, self.features_list_1]):
            if len(targets_indices) > 0:
                selected_inputs = inputs[targets_indices].to(device)
                with torch.no_grad(): 
                    features = self.feature_extractor(selected_inputs)
                    f5 = features[-1]
                    f5_flattened = torch.flatten(f5, start_dim=1).cpu()
                    features_list.append(f5_flattened)

    def _train_models(self, X_train):
        self.models = []
        k = self.num_PseudoLabeler

        if k > 1:
            kf = KFold(n_splits=k)
            for i, (_, index) in enumerate(kf.split(X_train)):
                X_train_fold = X_train[index]
                gaussian_density = GaussianDensityTorch()
                gaussian_density.fit(X_train_fold)
                self.models.append(gaussian_density)
                print(f"已训练 {i+1}/{k}")
        else:
            gaussian_density = GaussianDensityTorch()
            gaussian_density.fit(X_train)
            self.models.append(gaussian_density)
        print("GMM预测器训练完成")

    
    def _determine_thresholds(self, labeled_scores_list, unlabeled_scores_list=None):
        """
        确定阈值，针对正阈值（eta_p）需要两个得分列表：labeled_scores_list 和 unlabeled_scores_list。
        对于负阈值（eta_n），只需要一个得分列表：unlabeled_scores_list。
        """
        if unlabeled_scores_list is not None:
            # 处理正阈值eta_p
            assert len(labeled_scores_list) == len(unlabeled_scores_list), "Scores lists must have the same length"
            best_eta_list = []
            for labeled_scores, unlabeled_scores in zip(labeled_scores_list, unlabeled_scores_list):
                min_distance = np.inf
                best_eta = None
                for eta in np.linspace(0, 1, 100):
                    filtered_unlabeled_scores = unlabeled_scores[unlabeled_scores > eta]
                    if len(filtered_unlabeled_scores) > 0:
                        distance = wasserstein_distance(labeled_scores, filtered_unlabeled_scores)
                        if distance < min_distance:
                            min_distance = distance
                            best_eta = eta
                best_eta_list.append(best_eta)
        else:
            # 处理负阈值eta_n
            best_eta_list = []
            for unlabeled_scores in labeled_scores_list:  # 此处变量名的使用需要反映其作为未标记得分的用途
                unlabeled_scores = unlabeled_scores.cpu().numpy()
                eta_n = threshold_otsu(unlabeled_scores)
                eta_n = max(min(eta_n, 1), 0)
                best_eta_list.append(eta_n)

        return best_eta_list
    
    def get_model_scores(self):
        scores_list0 = [model.predict(self.X_train0) for model in self.models]
        scores_list1 = [model.predict(self.X_train1) for model in self.models]
        # print(scores_list0, scores_list1)
        scores_list0, scores_list1 = maxmin_score_normalize(scores_list0, scores_list1)
        # print(scores_list0, scores_list1)
        return scores_list0, scores_list1

    def fit(self, trainloader, device):
        self.features_list_0 = []
        self.features_list_1 = []
        for inputs, _, targets in trainloader:
            self._extract_features(inputs, targets, device)

        self.X_train0 = torch.cat(self.features_list_0, axis=0)
        self.X_train1 = torch.cat(self.features_list_1, axis=0)

        self._train_models(self.X_train0)

        labeled_scores_list, unlabeled_scores_list = self.get_model_scores()

        self.eta_p = self._determine_thresholds(labeled_scores_list, unlabeled_scores_list)
        self.eta_n = self._determine_thresholds(unlabeled_scores_list)
        print(self.eta_p, self.eta_n)

    # BCE损失函数对于有标签数据
    def bce_loss_labeled(self, x_l, y_l):
        # 计算BCE损失
        bce_loss = [F.binary_cross_entropy_with_logits(x, y) for x, y in zip(x_l, y_l)]
        average_loss = sum(bce_loss) / len(bce_loss)
        return average_loss
    
    # BCE损失函数对于伪标签数据
    def bce_loss_unlabeled(self, x_u_pre, x_u_pl, u_e):
        # 伪标签存在时才计算损失
        bce_loss = [F.binary_cross_entropy_with_logits(x, y, reduction='none') for x, y in zip(x_u_pre, x_u_pl)]
        bce_loss = torch.stack(bce_loss)
        average_loss = torch.mean(bce_loss)
        average_loss = average_loss * u_e
        return average_loss
    
    def generate_pseudo_labelers(self, unlabeled_scores_list):
        """
        根据来自单类别分类器（OCCs）的分数和确定的阈值（eta_p用于正伪标签生成器，eta_n用于负伪标签生成器），为未标记数据生成伪标签。

        参数：
        - unlabeled_scores_list：来自OCCs的未标记样本的分数列表。

        返回：
        - pseudo_labels_positive：由正伪标签生成器生成的伪标签。
        - pseudo_labels_negative：由负伪标签生成器生成的伪标签。
        """
        pseudo_labels_positive = []
        pseudo_labels_negative = []

        # 循环遍历每个样本在所有OCCs上的分数
        for scores in zip(*unlabeled_scores_list):
            # 生成正伪标签
            positive_votes = [score > eta for score, eta in zip(scores, self.eta_p)]
            # 只有当所有OCCs都投票为正时才标记为1，否则为0
            pseudo_label_positive = 1 if all(positive_votes) else 0
            pseudo_labels_positive.append(pseudo_label_positive)

            # 生成负伪标签
            negative_votes = [score <= eta for score, eta in zip(scores, self.eta_n)]
            # 只有当所有OCCs都投票为负（即分数低于阈值）时才标记为1，否则为0
            pseudo_label_negative = 1 if all(negative_votes) else 0
            pseudo_labels_negative.append(pseudo_label_negative)

        return pseudo_labels_positive, pseudo_labels_negative

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def normalize_scores(scores_list):
    normalized_scores_list = []
    for scores in scores_list:
        normalized_scores = sigmoid(scores)
        normalized_scores_list.append(normalized_scores)
    return normalized_scores_list

def z_score_normalize(scores_list0, scores_list1):
    normalized_scores_list0 = []
    normalized_scores_list1 = []
    for scores0, scores1 in zip(scores_list0, scores_list1):
        combined_scores = torch.cat((scores0, scores1), dim=0)
        mean = combined_scores.mean()
        std = combined_scores.std()
        normalized_combined_scores = (combined_scores - mean) / std
        normalized_normal_scores = normalized_combined_scores[:len(scores0)]
        normalized_unlabeled_scores = normalized_combined_scores[len(scores0):]
        normalized_scores_list0.append(normalized_normal_scores)
        normalized_scores_list1.append(normalized_unlabeled_scores)
    return normalized_scores_list0, normalized_scores_list1

def maxmin_score_normalize(scores_list0, scores_list1):
    normalized_scores_list0 = []
    normalized_scores_list1 = []
    for scores0, scores1 in zip(scores_list0, scores_list1):
        combined_scores = torch.cat((scores0, scores1), dim=0)
        min_val = combined_scores.min()
        max_val = combined_scores.max()
        normalized_combined_scores = (combined_scores - min_val) / (max_val - min_val)
        normalized_normal_scores = normalized_combined_scores[:len(scores0)]
        normalized_unlabeled_scores = normalized_combined_scores[len(scores0):]
        normalized_scores_list0.append(normalized_normal_scores)
        normalized_scores_list1.append(normalized_unlabeled_scores)
    return normalized_scores_list0, normalized_scores_list1
