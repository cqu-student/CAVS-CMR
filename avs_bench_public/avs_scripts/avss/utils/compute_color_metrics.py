import torch
from torch.nn import functional as F

import os
import shutil
import logging
import cv2
import numpy as np
from PIL import Image
import itertools
import sys
import time
import pandas as pd
from torchvision import transforms
import json
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, 'tasks'))
import get_tasks
import pdb

#根据列表转换为对应的类别
def transfer(task_list):
    with open('../../avsbench_data/AVSBench_semantic/label2idx.json', 'r') as f:
        categories = json.load(f)
    # 创建映射字典，注意 JSON 文件中的数字比实际的要大一个
    mapping_dict = {int(value) - 1: key for key, value in categories.items()}
    named_list = [mapping_dict[num] for num in task_list]
    return named_list

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()   
    
class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes, dataset, cfg):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        #ToDO: 搞一个从tasks映射过来的
        if cfg.dataset_name == 'v2s':
            task_list = list(itertools.chain(*get_tasks.task_v2s[cfg.task_name].values()))
            self.CLASSES = transfer(task_list)
        elif cfg.dataset_name == 'v2m':
            task_list = list(itertools.chain(*get_tasks.task_v2m[cfg.task_name].values()))
            self.CLASSES = transfer(task_list)
        elif cfg.dataset_name == 'v2':
            task_list = list(itertools.chain(*get_tasks.task_v2[cfg.task_name].values()))
            self.CLASSES = transfer(task_list)
        else:
            NotImplementedError
        
    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
    
    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU" and k!="Class Acc":
                string += "%s: %f\n"%(k, v)
        
        string+='Class IoU/Acc:\n'
        for (k, v1), v2 in zip(results['Class IoU'].items(), results['Class Acc'].values()):
            string += "\%s: %.4f (miou) , %.4f (acc) \n" % (self.CLASSES[k], v1, v2)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        # 乘 n 是为了转到正确的位置上
        hist = torch.bincount(
            self.n_classes * label_true[mask].long() + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        EPS = 1e-6
        hist = self.confusion_matrix
        acc = torch.diag(hist).sum() / hist.sum()
        acc_cls = torch.diag(hist) / (hist.sum(dim=1) + EPS)
        cls_acc = dict(zip(range(self.n_classes), acc_cls.cpu().numpy()))  # 转回 CPU 用于输出
        acc_cls = torch.nanmean(acc_cls)
        
        iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist) + EPS)
        mean_iu = torch.nanmean(iu)
        freq = hist.sum(dim=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu.cpu().numpy()))  # 转回 CPU 用于输出

        return {
                "Overall Acc": acc.item(),
                "Mean Acc": acc_cls.item(),
                "FreqW Acc": fwavacc.item(),
                "Class Acc": cls_acc,
                "Mean IoU": mean_iu.item(),
                "Class IoU": cls_iu,
            }
        
    def reset(self):
        self.confusion_matrix = torch.zeros((self.n_classes, self.n_classes), device='cuda')

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]


def _batch_miou_fscore(output, target, nclass, T, beta2=0.3):
    """batch mIoU and Fscore"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1
    # pdb.set_trace()
    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    fscores = torch.zeros(nclass).float()

    # vid_miou_list = torch.zeros(target.shape[0]).float()
    vid_miou_list = []
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi) # TP
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FP
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi) # TP + FN
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        # iou[torch.isnan(iou)] = 1.
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1

        precision = area_inter / area_pred
        recall = area_inter / area_lab
        fscore = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        fscore[torch.isnan(fscore)] = 0.
        fscores += fscore

        vid_miou_list.append(torch.sum(iou) / (torch.sum( iou != 0 ).float()))

    return ious, fscores, cls_count, vid_miou_list


def calc_color_miou_fscore(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """  
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # miou, fscore, cls_count = _batch_miou_fscore(pred, target, nclass, T) 
    miou, fscore, cls_count, vid_miou_list = _batch_miou_fscore(pred, target, nclass, T) 
    return miou, fscore, cls_count, vid_miou_list


def _batch_intersection_union(output, target, nclass, T):
    """mIoU"""
    # output: [BF, C, H, W],
    # target: [BF, H, W]
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    # pdb.set_trace()

    predict = predict.float() * (target > 0).float() # [BF, H, W]
    intersection = predict * (predict == target).float() # [BF, H, W]
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    batch_size = target.shape[0] // T
    cls_count = torch.zeros(nclass).float()
    ious = torch.zeros(nclass).float()
    for i in range(target.shape[0]):
        area_inter = torch.histc(intersection[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target[i].cpu(), bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        iou = 1.0 * area_inter.float() / (2.220446049250313e-16 + area_union.float())
        ious += iou
        cls_count[torch.nonzero(area_union).squeeze(-1)] += 1
        # pdb.set_trace()
    # ious = ious / cls_count
    # ious[torch.isnan(ious)] = 0
    # pdb.set_trace()
    # return area_inter.float(), area_union.float()
    # return ious
    return ious, cls_count


def calc_color_miou(pred, target, T=10):
    r"""
    J measure
        param: 
            pred: size [BF x C x H x W], C is category number including background
            target: size [BF x H x W]
    """  
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    # correct, labeled = _batch_pix_accuracy(pred, target)
    # inter, union = _batch_intersection_union(pred, target, nclass, T)
    ious, cls_count = _batch_intersection_union(pred, target, nclass, T)

    # pixAcc = 1.0 * correct / (2.220446049250313e-16 + labeled)
    # IoU = 1.0 * inter / (2.220446049250313e-16 + union)
    # mIoU = IoU.mean().item()
    # pdb.set_trace()
    # return mIoU
    return ious, cls_count


def calc_binary_miou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x C x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # assert len(pred.shape) == 3 and pred.shape == target.shape
    nclass = pred.shape[1]
    pred = torch.softmax(pred, dim=1) # [BF, C, H, W]
    pred = torch.argmax(pred, dim=1) # [BF, H, W]
    binary_pred = (pred != (nclass - 1)).float() # [BF, H, W]
    # pdb.set_trace()
    pred = binary_pred
    target = (target != (nclass - 1)).float()

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union+eps)) / N
    # pdb.set_trace()
    return iou



if __name__ == "__main__":
    print("done")
    pred = torch.ones(5, 10, 10)
    pred[:, :5, :5] = 0
    pred[:, :]
    label = torch.ones(5, 10, 10)

