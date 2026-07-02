import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging

from config import cfg
from color_dataloader import V2Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss
from loss import IcarlLoss, BCEWithLogitsLossWithIgnoreIndex, UnbiasedKnowledgeDistillationLoss, KnowledgeDistillationLoss
from regularizer import get_regularizer
from utils import pyutils
from utils.utility import logger
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.system import setup_logging
import pdb
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, 'tasks'))
import torchvision
import get_tasks
import pickle
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
kwargs = DDPK(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], mixed_precision="fp16")

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

def save_ckpt(path, model, optimizer, regularizer):
    if regularizer == None:
        state_dict={
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'regularizer_state' : None,
    }
    else:
        state_dict={
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'regularizer_state' : regularizer.state_dict(),
        }
    torch.save(state_dict, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="AVSS", type=str, help="the AVSS setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--val_batch_size", default=2, type=int)
    parser.add_argument("--max_epoches", default=60, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--start_eval_epoch", default=15, type=int)
    parser.add_argument("--eval_interval", default=2, type=int)

    parser.add_argument('--masked_av_flag', action='store_true', default=False, help='additional sa/masked_va loss for five frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing loss')
    parser.add_argument("--masked_av_stages", default=[], nargs='+', type=int, help='compute sa/masked_va loss in which stages: [0, 1, 2, 3]')
    parser.add_argument('--threshold_flag', action='store_true', default=False, help='whether thresholding the generated masks')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')
    parser.add_argument('--norm_fea_flag', action='store_true', default=False, help='normalize audio-visual features')
    parser.add_argument('--closer_flag', action='store_true', default=False, help='use closer loss for masked_va loss')
    parser.add_argument('--euclidean_flag', action='store_true', default=False, help='use euclidean distance for masked_va loss')
    parser.add_argument('--kl_flag', action='store_true', default=False, help='use kl loss for masked_va loss')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3]')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")
    
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # checkpoint_dir = os.path.join('./checkpoints', cfg.dataset_name, cfg.task_name, str(cfg.step))
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    # args.checkpoint_dir = checkpoint_dir
    # cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step-1)
    # model_old = AVSModel.Pred_endecoder(channel=256, \
    #                                     config=cfg, \
    #                                 tpavi_stages=args.tpavi_stages, \
    #                                 tpavi_vv_flag=args.tpavi_vv_flag, \
    #                                 tpavi_va_flag=args.tpavi_va_flag)
    # step_checkpoint = torch.load(cfg.step_checkpoint, map_location='cpu')
    # regularizer_state = step_checkpoint['regularizer_state']
    # # pdb.set_trace()
    # model_old.load_state_dict(step_checkpoint['model_state_dict'], strict=True)
    # model_old = accelerator.prepare(model_old)
    # for par in model_old.parameters():
    #     par.requires_grad = False
    # model_old.eval()

    # Data
    train_dataset = V2Dataset('train') 
    # train_dataset = V2Dataset('train', debug_flag=True) 
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = V2Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    
    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio_path, label, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data
        vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
        gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()
        imgs = imgs.cuda()
        label = label.cuda()
        B, frame, C, H, W = imgs.shape
        imgs = imgs.view(B*frame, C, H, W)
        mask_num = 10
        label = label.view(B*mask_num, H, W)
        audio_feature = []
        for audio_path_ in audio_path:
            with open(audio_path_, 'rb') as f:
                audio_feature1 = pickle.load(f)
            audio_feature.append(audio_feature1)
        audio_feature = torch.cat(audio_feature, dim=0).cuda()
        #! notice
        vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
        gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]
        print(audio_path)
        for i in range(len(audio_path)):
            save_path = os.path.join(os.path.dirname(audio_path[i]),"60-10.pkl")
            if os.path.exists(save_path):
                os.remove(save_path)