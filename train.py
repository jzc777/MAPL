import time
import json
import os 
import wandb
import logging

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from sklearn.metrics import roc_auc_score
from metrics import compute_pro, trapezoid
import pandas as pd

# 在函数开始处初始化一个空列表来收集数据
training_records = []


_logger = logging.getLogger('train')

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def training(model, trainloader, validloader, criterion, optimizer, scheduler,pseudo_labeler, epochs, num_training_steps: int = 1000, loss_weights: List[float] = [0.6, 0.4],
             log_interval: int = 1, eval_interval: int = 1, savedir: str = None, use_wandb: bool = False, device: str ='cpu') -> dict:   

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    l1_losses_m = AverageMeter()
    focal_losses_m = AverageMeter()
    final_eval_metrics = {}
    best_score_info = {}

    # criterion
    l1_criterion, focal_criterion = criterion
    l1_weight, focal_weight = loss_weights
    
    # set train mode
    model.train()

    # set optimizer
    optimizer.zero_grad()

    # training
    best_score = 0
    step = 0
    train_mode = True

    epoch = 0

    while train_mode:

        print("进行第{}轮更新".format(epoch + 1))

        end = time.time()
        pseudo_labeler.fit(trainloader, device)
        labeled_scores_list, unlabeled_scores_list = pseudo_labeler.get_model_scores()
        pseudo_labels_positive, pseudo_labels_negative = pseudo_labeler.generate_pseudo_labelers(unlabeled_scores_list)
        pseudo_labels_positive = [torch.full_like(x, fill_value=y) for x, y in zip(unlabeled_scores_list, pseudo_labels_positive)]
        pseudo_labels_negative = [torch.full_like(x, fill_value=y) for x, y in zip(unlabeled_scores_list, pseudo_labels_negative)]

        # print(pseudo_labels_positive, pseudo_labels_negative)

        truth_labels = [torch.ones_like(x) for x in labeled_scores_list]

        total_l_labeled = 0
        total_l_unlabeled_p = 0
        total_l_unlabeled_n = 0
        num_models = pseudo_labeler.num_PseudoLabeler

        for i in range(num_models):
            l_labeled = pseudo_labeler.bce_loss_labeled(labeled_scores_list, truth_labels)
            l_unlabeled_p = pseudo_labeler.bce_loss_unlabeled(unlabeled_scores_list, pseudo_labels_positive, 1)
            l_unlabeled_n = pseudo_labeler.bce_loss_unlabeled(unlabeled_scores_list, pseudo_labels_negative, 1)

            total_l_labeled += l_labeled
            total_l_unlabeled_p += l_unlabeled_p
            total_l_unlabeled_n += l_unlabeled_n

        avg_l_labeled = (total_l_labeled / num_models).to(device)
        avg_l_unlabeled_p = (total_l_unlabeled_p / num_models).to(device)
        avg_l_unlabeled_n = (total_l_unlabeled_n / num_models).to(device)

        print(avg_l_labeled, avg_l_unlabeled_p, avg_l_unlabeled_n)


        for inputs, masks, targets in trainloader:

            # batch
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)

            data_time_m.update(time.time() - end)

            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            l1_loss = l1_criterion(outputs[:,1,:], masks)
            focal_loss = focal_criterion(outputs, masks)
            loss = (l1_weight * l1_loss) + (focal_weight * focal_loss)

            loss = loss + avg_l_labeled + avg_l_unlabeled_p

            loss.backward()
            
            # update weight
            optimizer.step()
            optimizer.zero_grad()

            # log loss
            l1_losses_m.update(l1_loss.item())
            focal_losses_m.update(focal_loss.item())
            losses_m.update(loss.item())
            
            batch_time_m.update(time.time() - end)

            # wandb
            if use_wandb:
                wandb.log({
                    'lr':optimizer.param_groups[0]['lr'],
                    'train_focal_loss':focal_losses_m.val,
                    'train_l1_loss':l1_losses_m.val,
                    'train_loss':losses_m.val
                },
                step=step)
            
            if (step+1) % log_interval == 0 or step == 0: 
                _logger.info('TRAIN [{:>4d}/{}] '
                            'Loss: {loss.val:>6.4f} ({loss.avg:>6.4f}) '
                            'L1 Loss: {l1_loss.val:>6.4f} ({l1_loss.avg:>6.4f}) '
                            'Focal Loss: {focal_loss.val:>6.4f} ({focal_loss.avg:>6.4f}) '
                            'L_BCE Loss: {l_labeled} '
                            'unL_BCE Loss: {l_unlabeled} ' 
                            'LR: {lr:.3e} '
                            'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                            'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            step+1, num_training_steps, 
                            loss       = losses_m, 
                            l1_loss    = l1_losses_m,
                            focal_loss = focal_losses_m,
                            l_labeled  = avg_l_labeled,
                            l_unlabeled= avg_l_unlabeled_p,
                            lr         = optimizer.param_groups[0]['lr'],
                            batch_time = batch_time_m,
                            rate       = inputs.size(0) / batch_time_m.val,
                            rate_avg   = inputs.size(0) / batch_time_m.avg,
                            data_time  = data_time_m))


            if ((step+1) % eval_interval == 0 and step != 0) or (step+1) == num_training_steps: 
                eval_metrics = evaluate(
                    model        = model, 
                    dataloader   = validloader, 
                    device       = device
                )
                model.train()

                eval_log = dict([(f'eval_{k}', v) for k, v in eval_metrics.items()])

                # wandb
                if use_wandb:
                    wandb.log(eval_log, step=step)

                # checkpoint
                if best_score < np.mean(list(eval_metrics.values())):
                    # save best score
                    state = {'best_step':step}
                    state.update(eval_log)
                    json.dump(state, open(os.path.join(savedir, 'best_score.json'),'w'), indent='\t')

                    # save best model
                    torch.save(model.state_dict(), os.path.join(savedir, f'best_model.pt'))
                    
                    _logger.info('Best Score {0:.3%} to {1:.3%}'.format(best_score, np.mean(list(eval_metrics.values()))))

                    best_score = np.mean(list(eval_metrics.values()))

                eval_metrics = evaluate(model, validloader, device)
                final_eval_metrics = eval_metrics.copy()  # 更新最终的评估结果

                # 更新最佳分数信息
                current_score = np.mean(list(eval_metrics.values()))
                if best_score < current_score:
                    best_score = current_score
                    best_score_info = {
                        'Best Score': best_score,
                        'Step': step + 1
                    }

            # scheduler
            if scheduler:
                scheduler.step()

            end = time.time()

            step += 1
            if step == num_training_steps:
                train_mode = False
                break

            # 在日志记录部分，将数据添加到列表中
            training_records.append({
                'Step': step + 1,
                'Loss': losses_m.val,
                'L1 Loss': l1_losses_m.val,
                'Focal Loss': focal_losses_m.val,
                'LR': optimizer.param_groups[0]['lr'],
                'Time': batch_time_m.val,
                'Data Time': data_time_m.val
            })

            if step == num_training_steps:
                train_mode = False
                break

        epoch = epoch + 1
        if epoch == epochs:
            break

    # print best score and step
    _logger.info('Best Metric: {0:.3%} (step {1:})'.format(best_score, state['best_step']))

    # save latest model
    torch.save(model.state_dict(), os.path.join(savedir, f'latest_model.pt'))

    # save latest score
    state = {'latest_step':step}
    state.update(eval_log)
    json.dump(state, open(os.path.join(savedir, 'latest_score.json'),'w'), indent='\t')

    # 训练结束后，将数据转换为DataFrame并写入Excel文件
    df = pd.DataFrame(training_records)

    final_records = [
        {'Metric': 'AUROC-image', 'Value': final_eval_metrics.get('AUROC-image', 0)},
        {'Metric': 'AUROC-pixel', 'Value': final_eval_metrics.get('AUROC-pixel', 0)},
        {'Metric': 'AUPRO-pixel', 'Value': final_eval_metrics.get('AUPRO-pixel', 0)},
        {'Metric': 'Best Score', 'Value': best_score_info.get('Best Score', 0), 'Step': best_score_info.get('Step', 0)}
    ]

    excel_path = os.path.join(savedir, 'training_records.xlsx')
    df.to_excel(excel_path, index=False)

    df1 = pd.DataFrame(final_records)
    excel_path = os.path.join(savedir, 'final_evaluation_records.xlsx')
    df1.to_excel(excel_path, index=False)


def evaluate(model, dataloader, device: str = 'cpu'):
    # targets and outputs
    image_targets = []
    image_masks = []
    anomaly_score = []
    anomaly_map = []


    model.eval()
    with torch.no_grad():
        for idx, (inputs, masks, targets) in enumerate(dataloader):
            inputs, masks, targets = inputs.to(device), masks.to(device), targets.to(device)
            
            # predict
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            anomaly_score_i = torch.topk(torch.flatten(outputs[:,1,:], start_dim=1), 100)[0].mean(dim=1)

            # stack targets and outputs
            image_targets.extend(targets.cpu().tolist())
            image_masks.extend(masks.cpu().numpy())
            
            anomaly_score.extend(anomaly_score_i.cpu().tolist())
            anomaly_map.extend(outputs[:,1,:].cpu().numpy())
            
    # metrics    
    image_masks = np.array(image_masks)
    anomaly_map = np.array(anomaly_map)
    
    auroc_image = roc_auc_score(image_targets, anomaly_score)
    auroc_pixel = roc_auc_score(image_masks.reshape(-1).astype(int), anomaly_map.reshape(-1))
    all_fprs, all_pros = compute_pro(
        anomaly_maps      = anomaly_map,
        ground_truth_maps = image_masks
    )
    aupro = trapezoid(all_fprs, all_pros)
    
    metrics = {
        'AUROC-image':auroc_image,
        'AUROC-pixel':auroc_pixel,
        'AUPRO-pixel':aupro

    }



    _logger.info('TEST: AUROC-image: %.3f%% | AUROC-pixel: %.3f%% | AUPRO-pixel: %.3f%%' % 
                (metrics['AUROC-image'], metrics['AUROC-pixel'], metrics['AUPRO-pixel']))


    return metrics
