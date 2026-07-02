import math
import os
import time
import numpy as np
import torch
import argparse
from torch.utils import data
import pandas as pd
from color_dataloader import V2Dataset, load_image_in_PIL_to_Tensor
from config import cfg
import get_tasks
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from utils.compute_color_metrics import _batch_intersection_union
import pickle
import json
from torchvision import transforms
from PIL import Image
from utils.vis_mask import save_color_mask
from color_dataloader import get_v2_pallete
import torchvision
kwargs = DDPK(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="AVSS", type=str, help="the AVSS setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--test_batch_size", default=2, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add non-local block in which stages: [0, 1, 2, 3')

    args = parser.parse_args()
    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")
    cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name, cfg.task_name, cfg.step)
    model_vv = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=True, \
                                        tpavi_va_flag=False)
    step_checkpoint_v = torch.load("checkpoints/v2/60-10/0/miou_best_vv.pth", map_location='cpu')
    model_vv.load_state_dict(step_checkpoint_v['model_state_dict'])
    model_vv = accelerator.prepare(model_vv)

    model_va = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=False, \
                                        tpavi_va_flag=True)
    step_checkpoint_va = torch.load("checkpoints/v2/60-10/0/miou_best.pth", map_location = 'cpu')
    model_va.load_state_dict(step_checkpoint_va['model_state_dict'])
    model_va = accelerator.prepare(model_va)
    for par in model_vv.parameters():
        par.requires_grad = False
    model_vv.eval()
    for par in model_va.parameters():
        par.requires_grad = False
    model_va.eval()
    with torch.no_grad():
        video_name, set = "73QQbJIeB3Y_460000_470000", "v2"
        img_base_path =  os.path.join(cfg.DATA.DIR_BASE, set, video_name, 'frames')
        audio_path = os.path.join(cfg.DATA.DIR_BASE, set, video_name, 'audio1.pkl')
        color_mask_base_path = os.path.join(cfg.DATA.DIR_BASE, set, video_name, 'labels_rgb')

        if set == 'v1s': # data from AVSBench-object single-source subset (5s, gt is only the first annotated frame)
            vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
            gt_temporal_mask_flag  = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])#.bool()
        elif set == 'v1m': # data from AVSBench-object multi-sources subset (5s, all 5 extracted frames are annotated)
            vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
            gt_temporal_mask_flag  = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
        elif set == 'v2': # data from newly collected videos in AVSBench-semantic (10s, all 10 extracted frames are annotated))
            vid_temporal_mask_flag = torch.ones(10)#.bool()
            gt_temporal_mask_flag = torch.ones(10)#.bool()

        img_path_list = sorted(os.listdir(img_base_path)) # 5 for v1, 10 for new v2
        imgs_num = len(img_path_list)
        imgs_pad_zero_num = 10 - imgs_num
        imgs = []
        for img_id in range(imgs_num):
            img_path = os.path.join(img_base_path, "%d.jpg"%(img_id))
            img = load_image_in_PIL_to_Tensor(img_path, split="train", transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]))
            imgs.append(img)
        for pad_i in range(imgs_pad_zero_num): #! pad black image?
            img = torch.zeros_like(img)
            imgs.append(img)

        imgs_tensor = torch.stack(imgs, dim=0)
        #! notice:
        # audio = audio.cuda()
        # mask = mask.cuda()
        frame, C, H, W = imgs_tensor.shape
        imgs_tensor = imgs_tensor.view(frame, C, H, W)
        # mask = mask.view(B*frame, H, W)
        #! notice
        vid_temporal_mask_flag = vid_temporal_mask_flag.view(frame) # [B*T]
        gt_temporal_mask_flag  = gt_temporal_mask_flag.view(frame)  # [B*T]
        with open(audio_path, 'rb') as f:
            audio_feature = pickle.load(f)
        imgs_tensor = imgs_tensor.cuda()
        audio_feature = audio_feature.cuda()
        vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
        output_v, _, _, _, _= model_vv(imgs_tensor, audio_feature, vid_temporal_mask_flag)
        output_va, _, _, _, _ = model_va(imgs_tensor, audio_feature, vid_temporal_mask_flag)
    # 提取第9张图的输出
    preds = output_va.detach().max(dim=1)[1]
    v2_pallete = get_v2_pallete(cfg.DATA.LABEL_IDX_PATH)
    resize_pred_mask = cfg.DATA.RESIZE_PRED_MASK
    if resize_pred_mask:
        pred_mask_img_size = cfg.DATA.SAVE_PRED_MASK_IMG_SIZE
    else:
        pred_mask_img_size = cfg.DATA.IMG_SIZE
    mask_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pred_masks')
    save_color_mask(output_v, preds, mask_save_path, ["22222"], v2_pallete, resize_pred_mask, pred_mask_img_size, T=10)