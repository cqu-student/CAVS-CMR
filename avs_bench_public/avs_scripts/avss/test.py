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

from utils import pyutils
from utils.utility import logger 
from utils.system import setup_logging

from utils.vis_mask import save_color_mask
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.compute_color_metrics import StreamSegMetrics
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, 'tasks'))
import torchvision
import get_tasks
import pickle
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate import load_checkpoint_and_dispatch
kwargs = DDPK(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

import pdb

import sys
sys.path.append('../')
from color_dataloader import get_v2_pallete


class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea


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
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir, '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S_'), np.random.randint(1, 10)))
    args.log_dir = log_dir

    # # Save scripts
    # script_path = os.path.join(log_dir, 'scripts')
    # if not os.path.exists(script_path):
    #     os.makedirs(script_path, exist_ok=True)
    
    # scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'color_dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    # for script in scripts_to_save:
    #     dst_path = os.path.join(script_path, script)
    #     try:
    #         shutil.copy(script, dst_path)
    #     except IOError:
    #         os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #         shutil.copy(script, dst_path)

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step)


    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    # model = load_checkpoint_and_dispatch(
    #             model,  # 你的模型
    #             cfg.step_checkpoint,  # 检查点路径
    #             device_map="auto",  # 自动分配设备
    #         )
    step_checkpoint = torch.load(cfg.step_checkpoint)
    model.load_state_dict(step_checkpoint['model_state_dict'])
    model = accelerator.prepare(model)
    logger.info('Load trained model %s'%cfg.step_checkpoint)

    # Test data
    split = 'test'
    test_dataset = V2Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)
    test_dataloader = accelerator.prepare(test_dataloader)
    N_CLASSES = sum(get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step))

    # Test
    model.eval()

    # for save predicted rgb masks
    v2_pallete = get_v2_pallete(cfg.DATA.LABEL_IDX_PATH)
    resize_pred_mask = cfg.DATA.RESIZE_PRED_MASK
    if resize_pred_mask:
        pred_mask_img_size = cfg.DATA.SAVE_PRED_MASK_IMG_SIZE
    else:
        pred_mask_img_size = cfg.DATA.IMG_SIZE

    metrics = StreamSegMetrics(N_CLASSES, dataset=cfg.dataset_name, cfg = cfg)
    metrics.reset()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio_path, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]

            #! notice:
            vid_temporal_mask_flag = vid_temporal_mask_flag
            gt_temporal_mask_flag  = gt_temporal_mask_flag

            imgs = imgs.cuda()
            # audio = audio.cuda()
            mask = mask
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W).cuda()
            mask = mask.view(B*frame, H, W)
    
            #! notice
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]

            audio_feature = []
            for audio_path_ in audio_path:
                with open(audio_path_, 'rb') as f:
                    audio_feature1 = pickle.load(f)
                audio_feature.append(audio_feature1)
            audio_feature = torch.cat(audio_feature, dim=0)
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            audio_feature = audio_feature.cuda()
            output, _, _, _, _= model(imgs, audio_feature, vid_temporal_mask_flag) # [5, 24, 224, 224] = [bs=1 * T=5, 24, 224, 224]
            # if args.save_pred_mask:
            #     if accelerator.is_main_process:
            #         mask_save_path = os.path.join(log_dir, 'pred_masks')
            #         save_color_mask(output, mask, mask_save_path, video_name_list, v2_pallete, resize_pred_mask, pred_mask_img_size, T=10)
            #     accelerator.wait_for_everyone()
            preds = output.detach().max(dim=1)[1]
            targets = mask.cuda()
            metrics.update(targets, preds)
        test_score = metrics.get_results()
        if accelerator.is_main_process:
            print(metrics.to_str(test_score))
            class_iou = list(test_score['Class IoU'].values())
            class_acc = list(test_score['Class Acc'].values())
            first_cls = len(get_tasks.get_tasks_(cfg.dataset_name, cfg.task_name, 0))
            logger.info(f"mean_iou: %.6f" % test_score['Mean IoU'])
            logger.info(f"...from 0 to {first_cls} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
            logger.info(f"...from {first_cls} to {len(class_iou)} : best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
            logger.info(f"...from 0 to {first_cls} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
            logger.info(f"...from {first_cls} to {len(class_iou)} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))
        accelerator.wait_for_everyone()





