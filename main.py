
import logging
import os
import torch
import torch.nn as nn
import argparse
import random
import sys

from omegaconf import OmegaConf
from timm import create_model
from data import create_dataset, create_dataloader
from models import MAPL, MemoryBank,Encoder,density
from focal_loss import FocalLoss
from train import training
from log import setup_default_logging
from utils import torch_seed
from scheduler import CosineAnnealingWarmupRestarts
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.density import GaussianDensitySklearn, GaussianDensityTorch
from sklearn.model_selection import KFold

from models.PL import PseudoLabeler

_logger = logging.getLogger('train')


def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg.SEED)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    _logger.info('Device: {}'.format(device))

    # savedir
    cfg.EXP_NAME = cfg.EXP_NAME + f"-{cfg.DATASET.target}"
    savedir = os.path.join(cfg.RESULT.savedir, cfg.EXP_NAME)
    os.makedirs(savedir, exist_ok=True)

    
    # wandb
    if cfg.TRAIN.use_wandb:
        wandb.init(name=cfg.EXP_NAME, project='MAPL', config=OmegaConf.to_container(cfg))

    # build datasets
    _logger.info("开始trainset加载")
    trainset = create_dataset(
        datadir                = cfg.DATASET.datadir,
        target                 = cfg.DATASET.target, 
        is_train               = True,
        resize                 = cfg.DATASET.resize,
        imagesize              = cfg.DATASET.imagesize,
        texture_source_dir     = cfg.DATASET.texture_source_dir,
        structure_grid_size    = cfg.DATASET.structure_grid_size,
        transparency_range     = cfg.DATASET.transparency_range,
        perlin_scale           = cfg.DATASET.perlin_scale,
        min_perlin_scale       = cfg.DATASET.min_perlin_scale,
        perlin_noise_threshold = cfg.DATASET.perlin_noise_threshold,
        use_mask               = cfg.DATASET.use_mask,
        bg_threshold           = cfg.DATASET.bg_threshold,
        bg_reverse             = cfg.DATASET.bg_reverse
    )

    _logger.info("开始testset加载")
    testset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = False,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )

    memoryset = create_dataset(
        datadir   = cfg.DATASET.datadir,
        target    = cfg.DATASET.target, 
        is_train  = True,
        to_memory = True,
        resize    = cfg.DATASET.resize,
        imagesize = cfg.DATASET.imagesize,
    )
    
    # build dataloader
    trainloader = create_dataloader(
        dataset     = trainset,
        train       = True,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )
    _logger.info("开始testloader加载")
    testloader = create_dataloader(
        dataset     = testset,
        train       = False,
        batch_size  = cfg.DATALOADER.batch_size,
        num_workers = cfg.DATALOADER.num_workers
    )

    memory_bank = MemoryBank(
        normal_dataset   = memoryset,
        nb_memory_sample = cfg.MEMORYBANK.nb_memory_sample,
        device           = device
    )

    _logger.info("开始特征提取")
    # build feature extractor
    feature_extractor = Encoder().to(device)

    memory_bank.update(feature_extractor=feature_extractor)
    torch.save(memory_bank, os.path.join(savedir, f'memory_bank.pt'))
    _logger.info('Update {} normal samples in memory bank'.format(cfg.MEMORYBANK.nb_memory_sample))
    
    num_PseudoLabeler = cfg.PSEUDOLABELER.num_PseudoLabeler
    pseudo_labeler = PseudoLabeler(num_PseudoLabeler, feature_extractor)
    # pseudo_labeler.fit(trainloader, feature_extractor, device)
    _logger.info("结束")   
    """
    features_list_0 = []
    features_list_1 = []
    for inputs, _, targets in trainloader:
        # _logger.info("进行1")
        # print(inputs.shape)
        # print(target)
        # inputs = inputs.to(device)
        # with torch.no_grad():
        #     features = feature_extractor(inputs)
        #     for i in features:
        #         print(i.shape)
        targets_indices0 = torch.where(targets == 0)[0]
        targets_indices1 = torch.where(targets == 1)[0]
        if len(targets_indices0) > 0:
            selected_inputs = inputs[targets_indices0].to(device)
            with torch.no_grad():  # 不需要计算梯度
                features = feature_extractor(selected_inputs)
                # 选择使用最后一层的特征，并将其平坦化
                # 假设我们使用最后一个特征图 f5，这里仅作为示例
                # _logger.info("进行2")
                f5 = features[-1]
                f5_flattened = torch.flatten(f5, start_dim=1).cpu()
                features_list_0.append(f5_flattened)
        if len(targets_indices1) > 0:
            selected_inputs = inputs[targets_indices1].to(device)
            with torch.no_grad():  # 不需要计算梯度
                features = feature_extractor(selected_inputs)
                # 选择使用最后一层的特征，并将其平坦化
                # 假设我们使用最后一个特征图 f5，这里仅作为示例
                # _logger.info("进行2")
                f5 = features[-1]
                f5_flattened = torch.flatten(f5, start_dim=1).cpu()
                features_list_1.append(f5_flattened)
    
    # 假设 feature 是包含所有特征数组的列表
    num_features0 = len(features_list_0)  # 获取特征数组的数量
    num_features1 = len(features_list_1) 
    print(num_features0, num_features1)

    _logger.info("开始2")

    X_train0 = torch.cat(features_list_0, axis=0)
    X_train1 = torch.cat(features_list_1, axis=0)

    k = 5
    kf = KFold(n_splits=k)
    
    models = []

    for _, index in kf.split(X_train0):
        print(index)

        X_train_fold = X_train0[index]
        save_dir = "saved_models/" + f"{cfg.DATASET.target}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"model_{index[0]}-{index[-1]}.pt")
        if os.path.exists(model_path):
            # 如果已存在模型文件，则加载模型
            gaussian_density = torch.load(model_path)
            print(f"Model {model_path} loaded")
        else:
            # 拟合模型
            gaussian_density = GaussianDensityTorch()
            gaussian_density.fit(X_train_fold)
            torch.save(gaussian_density, model_path)
            print(f"Model {model_path} saved")
        models.append(gaussian_density)
    
    train_scores0_list = [model.predict(X_train0) for model in models]
    train_scores1_list = [model.predict(X_train1) for model in models]
    print(train_scores0_list)
    print(train_scores1_list)

    from scipy.stats import wasserstein_distance
    from skimage.filters import threshold_otsu
    from sklearn.preprocessing import MinMaxScaler
    # 确定正类阈值 eta^p
    def determine_eta_p(labeled_scores, unlabeled_scores):
        best_eta_p_list = []
        for labeled, unlabeled in zip(labeled_scores, unlabeled_scores):
            best_eta_p = None
            min_distance = np.inf
            for eta in np.linspace(min(unlabeled), max(unlabeled), 100):
                filtered_unlabeled = unlabeled[unlabeled > eta]
                if len(filtered_unlabeled) > 0:
                    distance = wasserstein_distance(labeled, filtered_unlabeled)
                    if distance < min_distance:
                        min_distance = distance
                        best_eta_p = eta
            best_eta_p_list.append(best_eta_p)
        return best_eta_p_list

    # 确定负类阈值 eta^n 使用Otsu方法
    def determine_eta_n(unlabeled_scores):
        best_eta_n_list = []
        for unlabeled in unlabeled_scores:
            # 假设 unlabeled 是一个一维 numpy 数组，如果是 tensor 需要先转换
            if isinstance(unlabeled, torch.Tensor):
                unlabeled = unlabeled.numpy()
                
            # 归一化到 [0, 1] 区间
            unlabeled_min, unlabeled_max = unlabeled.min(), unlabeled.max()
            normalized_unlabeled = (unlabeled - unlabeled_min) / (unlabeled_max - unlabeled_min)
            
            # 直接使用 Otsu 方法找到阈值
            eta_n = threshold_otsu(normalized_unlabeled)
            
            # 将阈值映射回原始数据的范围
            eta_n_original = eta_n * (unlabeled_max - unlabeled_min) + unlabeled_min
            
            best_eta_n_list.append(eta_n_original)
            
        return best_eta_n_list

    # 调用函数确定阈值
    eta_p = determine_eta_p(train_scores0_list, train_scores1_list)
    eta_n = determine_eta_n(train_scores1_list)

    print("Positive thresholds eta_p:", eta_p)
    print("Negative thresholds eta_n:", eta_n)




    """

    _logger.info("数据加载完成，开始模型初始化")
    model = MAPL(
        memory_bank       = memory_bank,
        pseudo_labeler    = pseudo_labeler,
        feature_extractor = feature_extractor
    ).to(device)

    # Set training
    l1_criterion = nn.L1Loss()
    f_criterion = FocalLoss(
        gamma = cfg.TRAIN.focal_gamma, 
        alpha = cfg.TRAIN.focal_alpha
    )

    optimizer = torch.optim.AdamW(
        params       = filter(lambda p: p.requires_grad, model.parameters()), 
        lr           = cfg.OPTIMIZER.lr, 
        weight_decay = cfg.OPTIMIZER.weight_decay
    )

    epochs = cfg.TRAIN.epochs
    # 计算 T_max, 这里是一个例子，您需要根据实际情况来计算
    T_max = len(trainloader) * epochs  # 假设 trainloader 是您的数据加载器，epochs 是您计划的训练周期数

    # 初始化 CosineAnnealingLR 调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0001, last_epoch=-1, verbose=False)

    # Fitting model
    _logger.info("模型初始化完成，开始训练")
    training(
        model              = model, 
        epochs             = epochs,
        num_training_steps = cfg.TRAIN.num_training_steps, 
        trainloader        = trainloader, 
        validloader        = testloader, 
        criterion          = [l1_criterion, f_criterion], 
        loss_weights       = [cfg.TRAIN.l1_weight, cfg.TRAIN.focal_weight],
        optimizer          = optimizer,
        scheduler          = scheduler,
        pseudo_labeler     = pseudo_labeler,
        log_interval       = cfg.LOG.log_interval,
        eval_interval      = cfg.LOG.eval_interval,
        savedir            = savedir,
        device             = device,
        use_wandb          = cfg.TRAIN.use_wandb
    )



if __name__=='__main__':
    args = OmegaConf.from_cli()
    # load default config
    cfg = OmegaConf.load(args.configs)
    del args['configs']
    
    # merge config with new keys
    cfg = OmegaConf.merge(cfg, args)
    
    # target cfg
    target_cfg = OmegaConf.load(cfg.DATASET.anomaly_mask_info)
    cfg.DATASET = OmegaConf.merge(cfg.DATASET, target_cfg[cfg.DATASET.target])
    
    print(OmegaConf.to_yaml(cfg))

    run(cfg)
