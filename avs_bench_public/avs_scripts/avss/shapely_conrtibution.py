import math
import os
import time
import numpy as np
import torch
import argparse
from torch.utils import data
import pandas as pd
from color_dataloader import V2Dataset
from config import cfg
import get_tasks
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from utils.compute_color_metrics import _batch_intersection_union
import pickle
import json


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

    parser.add_argument("--save_pred_mask", action='store_true', default=True, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)

    args = parser.parse_args()

    # if (args.visual_backbone).lower() == "resnet":
    #     from model import ResNet_AVSModel as AVSModel
    #     print('==> Use ResNet50 as the visual backbone...')
    # elif (args.visual_backbone).lower() == "pvt":
    #     from model import PVT_AVSModel as AVSModel
    #     print('==> Use pvt-v2 as the visual backbone...')
    # else:
    #     raise NotImplementedError("only support the resnet50 and pvt-v2")

    # train_dataset = V2Dataset('test')
    # train_dataloader = torch.utils.data.DataLoader(train_dataset, 
    #                                          batch_size= args.test_batch_size, 
    #                                          shuffle = True, 
    #                                          num_workers = 1, 
    #                                          pin_memory = True, 
    #                                          persistent_workers = True)
    # train_dataloader = accelerator.prepare(train_dataloader)
    # cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name, cfg.task_name, cfg.step)
    # ################################################################
    # model_vv = AVSModel.Pred_endecoder(channel=256, \
    #                                     config=cfg, \
    #                                     tpavi_stages=args.tpavi_stages, \
    #                                     tpavi_vv_flag=True, \
    #                                     tpavi_va_flag=False)
    # step_checkpoint_v = torch.load(cfg.step_checkpoint_v, map_location='cuda:0')
    # model_vv.load_state_dict(step_checkpoint_v['model_state_dict'])

    # model_va = AVSModel.Pred_endecoder(channel=256, \
    #                                     config=cfg, \
    #                                     tpavi_stages=args.tpavi_stages, \
    #                                     tpavi_vv_flag=False, \
    #                                     tpavi_va_flag=True)
    # step_checkpoint_va = torch.load(cfg.step_checkpoint, map_location = 'cuda:0')
    # model_va.load_state_dict(step_checkpoint_va['model_state_dict'])
    # for par in model_vv.parameters():
    #     par.requires_grad = False
    # model_vv.eval()
    # for par in model_va.parameters():
    #     par.requires_grad = False
    # model_va.eval()
    # model_va,model_vv = accelerator.prepare(model_va, model_vv)
    ########################################
    # memory_csv = '../../avsbench_data/AVSBench_semantic/memory_all_60_5a_v2s.csv'
    # # memory_csv = f'../../avsbench_data/AVSBench_semantic/memory.csv'
    # N_CLASSES = sum(get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all = pd.read_csv('../../avsbench_data/AVSBench_semantic/metadata.csv', sep=',')
    memory_csv = "../../avsbench_data/AVSBench_semantic/metatest.csv"
    df_split = df_all[df_all['split'] == 'test']
    label_index, _ = get_tasks.get_task_labels(cfg.dataset_name, cfg.task_name, cfg.step)
    with open('../../avsbench_data/AVSBench_semantic/idx2label.json', 'r') as fr:
        index_to_label = json.load(fr)
    label_array = [index_to_label[str(index+1)] for index in label_index]
    df_split.loc[:,'a_obj_split'] = df_split['a_obj'].str.split('_')
    df_split = df_split[df_split['a_obj_split'].apply(lambda x: any(label in label_array for label in x))]
    # df_split['contribution'] = None
    # with torch.no_grad():
    #     for n_iter, batch_data in enumerate(train_dataloader):
    #         imgs, audio_path, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, video_name_list = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
    #         # import pdb;pdb.set_trace()
    #         #! notice:
    #         vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
    #         gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()

    #         imgs = imgs.cuda()
    #         # audio = audio.cuda()
    #         mask = mask.cuda()
    #         B, frame, C, H, W = imgs.shape
    #         imgs = imgs.view(B*frame, C, H, W)
    #         mask = mask.view(B*frame, H, W)
    
    #         #! notice
    #         vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
    #         gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]

    #         audio_feature = []
    #         for audio_path_ in audio_path:
    #             with open(audio_path_, 'rb') as f:
    #                 audio_feature1 = pickle.load(f)
    #             audio_feature.append(audio_feature1)
    #         audio_feature = torch.cat(audio_feature, dim=0).cuda()

    #         output_v, _, _, _, _= model_vv(imgs, audio_feature, vid_temporal_mask_flag) # [5, 24, 224, 224] = [bs=1 * T=5, 24, 224, 224]
    #         output_va, _, _, _, _ = model_va(imgs, audio_feature, vid_temporal_mask_flag)

    #         # preds_v = output_v.detach().max(dim=1)[1].cpu().numpy()
    #         # preds_va = output_va.detach().max(dim=1)[1].cpu().numpy()
    #         # targets = mask.cpu().numpy()
    #         iou_v, _ = _batch_intersection_union(output_v, mask,N_CLASSES, T=10)
    #         iou_va,_ = _batch_intersection_union(output_va,mask,N_CLASSES, T=10)
    #         contribution_v = (iou_va.sum() - iou_v.sum()).item()
    #         print("n_iter:{}, contribution_v:{}".format(n_iter, contribution_v))
    #         video_name = video_name_list[0]
    #         matching_rows = df_split.loc[df_split['uid'] == video_name]
    #         # import pdb;pdb.set_trace()
    #         df_split.loc[matching_rows.index[0],'contribution'] = contribution_v
    #         df_split.loc[matching_rows.index[0],'miou_v'] = iou_v.sum().item()
    #         df_split.loc[matching_rows.index[0],'miou_va'] = iou_va.sum().item()
    # df_split = df_split.dropna()
    # df_split['contribution'] = df_split['contribution'].astype(float)
    df_split.to_csv(memory_csv, index=False)



    
