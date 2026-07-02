import os
import time
import random
import shutil
import torch
import numpy as np
import argparse
import logging
import gc
from config import cfg, modify_command_options
from color_dataloader import V2Dataset
from torchvggish import vggish
from loss import IouSemanticAwareLoss
from loss import IcarlLoss, BCEWithLogitsLossWithIgnoreIndex, UnbiasedKnowledgeDistillationLoss, KnowledgeDistillationLoss, UnbiasedCrossEntropy, soft_crossentropy, FDA_loss
from regularizer import get_regularizer
from utils import pyutils
from utils.utility import logger
from utils.compute_color_metrics import calc_color_miou_fscore
from utils.compute_color_metrics import StreamSegMetrics
from utils.system import setup_logging
import pdb
import sys
import pickle
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, 'tasks'))
import torchvision
import get_tasks
from functools import reduce
import pandas as pd
import math
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.utils import DeepSpeedPlugin
from plop import entropy, features_distillation
# from accelerate.utils import get_active_deepspeed_plugin
import warnings
from seg_diff import find_different_pixel_pairs, memory_sampling

# 取消 FutureWarning 警告
warnings.filterwarnings("ignore", category=FutureWarning)

class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea

def save_ckpt(path, model, optimizer, regularizer):
    model = accelerator.unwrap_model(model)
    if regularizer == None:
        state_dict={
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'regularizer_state' : None,
    }
    else:
        state_dict={
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'regularizer_state' : regularizer.state_dict(),
        }
    torch.save(state_dict, path)

def find_median(cfg, model, device, logger, mode = "probability"):
        import pickle
        train_dataset = V2Dataset('train')
        # if cfg.memory:
        #     train_dataset_ = V2Dataset('train')
        #     memory_dataset = V2Dataset('memory')
        #     train_dataset = torch.utils.data.ConcatDataset([train_dataset_, memory_dataset]) 
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=min(args.num_workers, 2),  # Reduce workers for median calculation
                                                        pin_memory=True,
                                                        persistent_workers=False)  # Don't persist workers for this temporary loader
        train_dataloader = accelerator.prepare(train_dataloader)
        if mode == "entropy":
            max_value = torch.log(torch.tensor(cfg.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        if cfg.pseudo_nb_bins is not None:
            nb_bins = cfg.pseudo_nb_bins
        memory_class = []
        histograms = torch.zeros(cfg.nb_current_classes, nb_bins).long().to(device)
        for n_iter, batch_data in enumerate(train_dataloader): 
            imgs, audio_path, label, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            original_labels = label.clone()
            memory_sample = []
            #! notice:
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda()
            gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda()
            
            imgs = imgs.cuda()
            label = label.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask_num = 10
            label = label.view(B*mask_num, H, W)
            #! notice
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]
            
            # Optimized audio feature loading in find_median function
            audio_feature = []
            if len(audio_path) > 1:
                for audio_path_ in audio_path:
                    with open(audio_path_, 'rb') as f:
                        audio_feature1 = pickle.load(f)
                    audio_feature.append(audio_feature1)
                audio_feature = torch.cat(audio_feature, dim=0)
            else:
                with open(audio_path[0], 'rb') as f:
                    audio_feature = pickle.load(f)
            audio_feature = audio_feature.cuda(non_blocking=True)
            outputs_old, features_old , _,  _, _= model(imgs, audio_feature,vid_temporal_mask_flag)
            ###########################################################
            predict_old = torch.argmax(outputs_old, dim = 1)
            predict_old_chunks = torch.chunk(predict_old, args.train_batch_size, dim=0)
            label_chunks = torch.chunk(label, args.train_batch_size, dim=0)
            for batch in range(args.train_batch_size):
                result, ratio = find_different_pixel_pairs(predict_old_chunks[batch], label_chunks[batch])
                # pdb.set_trace()
                if ratio > 0.7:
                    memory_sample.append(result)
            if len(memory_sample):
                memory_class.append(memory_sample)
            ###########################################################
            mask_bg = label == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

            if n_iter % 10 == 0:
                logger.info(f"Median computing {n_iter}/{len(train_dataloader)}.")
        ##############################################################
        #将memory_class保存到文件中
        memory_class_string = str(memory_class)
        # 保存为文本文件
        if accelerator.is_main_process:
            with open('60-5_2_v2s.txt', 'w', encoding='utf-8') as file:
                file.write(memory_class_string)  # 直接写入一整行
        accelerator.wait_for_everyone()
        ##############################################################
        del train_dataloader
        gc.collect()
        thresholds = torch.zeros(cfg.nb_current_classes, dtype=torch.float32).to(device)  # zeros or ones? If old_model never predict a class it may be important
        logger.info("Approximating median")
        for c in range(cfg.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = cfg.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        if cfg.step_threshold is not None:
            cfg.threshold += cfg.step * cfg.step_threshold
        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="AVSS", type=str, help="the AVSS setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=4, type=int)
    parser.add_argument("--max_epoches", default=60, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=2, type=int)  # Reduced from 4 to 2 to lower CPU usage
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
    modify_command_options(cfg)
    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    if os.path.exists(log_dir):
        log_dir = os.path.join(args.log_dir, '{}_{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'), np.random.randint(1, 10)))
   
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

    # Checkpoints directory
    checkpoint_dir = os.path.join('./checkpoints', cfg.dataset_name,"pvt",cfg.task_name, str(cfg.step))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Data
    if cfg.memory:
        memory_dataset = V2Dataset('memory')
        train_dataset_ = V2Dataset('train')
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_, memory_dataset])
    else:
        train_dataset = V2Dataset('train') 
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        persistent_workers=True if args.num_workers > 0 else False,
                                                        prefetch_factor=2 if args.num_workers > 0 else 2)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = V2Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        persistent_workers=True if args.num_workers > 0 else False,
                                                        prefetch_factor=2 if args.num_workers > 0 else 2)
    # active_plugin = get_active_deepspeed_plugin(accelerator.state)
    # assert active_plugin is deepspeed_plugins["student"]
    cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name, cfg.task_name, cfg.step)
    cfg.nb_new_classes = cfg.classes[-1]
    cfg.total_classes = reduce(lambda a, b: a+b, cfg.classes)
    cfg.nb_current_classes = cfg.total_classes #total classes
    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    logger.info("==> Total params: %.2fM" % ( sum(p.numel() for p in model.parameters()) / 1e6))
    # Optimizer
    model_params = model.parameters()
    if model.scalar is not None:
        model_params = list(model_params) + [model.scalar]
    optimizer = torch.optim.Adam(model_params, args.lr)

    N_CLASSES = sum(cfg.classes)
    metrics = StreamSegMetrics(N_CLASSES, dataset=cfg.dataset_name, cfg = cfg)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_reg_loss = pyutils.AverageMeter('reg_loss')
    #################################
    if cfg.step == 0:
        kwargs = DDPK(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[kwargs])
        model_old = None
        regularizer_state = None
        model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)
        optimizer = accelerator.prepare(optimizer)
        l_reg = torch.tensor(0.)
        if cfg.regularizer == None:
            regularizer = None
        else:
            device = accelerator.device
            regularizer = get_regularizer(model, model_old, device, cfg, regularizer_state)
        regulizer_flag = regularizer is not None
        cfg.old_classes = 0
    else:
        kwargs = DDPK(find_unused_parameters=True)
        zero2_plugin = DeepSpeedPlugin(hf_ds_config=os.path.join(root_dir, 'zero2_config.json'))
        zero3_plugin = DeepSpeedPlugin(hf_ds_config=os.path.join(root_dir, 'zero3_config.json'))
        deepspeed_plugin = {"student": zero2_plugin, "teacher": zero3_plugin}
        accelerator = Accelerator(kwargs_handlers=[kwargs], deepspeed_plugins = deepspeed_plugin)
        step_checkpoint = torch.load(cfg.step_checkpoint)
        model.load_state_dict(step_checkpoint['model_state_dict'], strict=False)
        if cfg.init_balanced:
            model.init_new_classifier(accelerator.device)
        elif cfg.init_multimodal is not None:
            model.init_new_classifier_multimodal(accelerator.device, train_dataloader, cfg.init_multimodal)
        regularizer_state = step_checkpoint['regularizer_state']
        cfg.classes = get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step-1)
        cfg.old_classes = cfg.total_classes - cfg.nb_new_classes
        model_old = AVSModel.Pred_endecoder(channel=256, \
                                            config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
        model_old.load_state_dict(step_checkpoint['model_state_dict'], strict=True)
        model, train_dataloader, val_dataloader = accelerator.prepare(model, train_dataloader, val_dataloader)
                # 创建一个新的字典来存储带有前缀的参数
        accelerator.state.select_deepspeed_plugin("teacher")
        # model_old = model_old.to(accelerator.device)
        # xxx Regularizer (EWC, RW, PI)
        l_reg = torch.tensor(0.)
        if cfg.regularizer == None:
            regularizer = None
        else:
            device = accelerator.device
            regularizer = get_regularizer(model, model_old, device, cfg, regularizer_state)
        regulizer_flag = regularizer is not None
        for par in model_old.parameters():
            par.requires_grad = False
        model_old = accelerator.prepare(model_old)
        model_old.eval()
    curr_idx = [
        sum(len(get_tasks.get_tasks_(cfg.dataset_name, cfg.task_name, step)) for step in range(cfg.step)), 
        sum(len(get_tasks.get_tasks_(cfg.dataset_name, cfg.task_name, step)) for step in range(cfg.step+1))
    ]
    #################################
    lde = cfg.loss_de
    lde_flag = lde > 0 and model_old is not None
    lde_loss = torch.nn.MSELoss()

    lkd = cfg.loss_kd
    lkd_flag = lkd > 0 and model_old is not None

    icarl_combined = False
    icarl_only_dist = False
    if cfg.icarl:
        icarl_combined = not cfg.icarl_disjoint and model_old is not None
        icarl_only_dist = cfg.icarl_disjoint and model_old is not None

    if lkd_flag:
        avg_meter_lkd_loss = pyutils.AverageMeter('lkd_loss')
    if icarl_only_dist or icarl_combined:
        avg_meter_icarl_loss = pyutils.AverageMeter('icarl_loss')
    if lde_flag:
        avg_meter_lde_loss = pyutils.AverageMeter('lde_loss')
    if cfg.pod:
        avg_meter_pod_loss = pyutils.AverageMeter('pod_loss')
    if cfg.entropy_min>0:
        avg_meter_entropy_loss = pyutils.AverageMeter('entropy_loss')
    #################################
    # Train
    best_epoch = 0
    global_step = 0
    miou_list = []
    best_score = 0
    miou_noBg_list = []
    fscore_list, fscore_noBg_list = [], []
    max_fs, max_fs_noBg = 0, 0
    model.train()
    if cfg.pseudo is None:
        pass
    elif cfg.pseudo.split("_")[0] == "median" and cfg.step > 0:
        logger.info("Find median score")
        cfg.thresholds, _ = find_median(train_dataloader, device, logger)
    elif cfg.pseudo.split("_")[0] == "entropy" and cfg.step > 0:
        logger.info("Find median score")
        thresholds, max_entropy = find_median(cfg, model_old, accelerator.device, logger, mode="entropy")
        cfg.thresholds = thresholds
        cfg.max_entropy = max_entropy
    #memory_dataset = V2Dataset('memory')
    # accelerator.wait_for_everyone()
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            imgs, audio_path, label, vid_temporal_mask_flag, gt_temporal_mask_flag, _ = batch_data # [bs, 5, 3, 224, 224], ->[bs, 5, 1, 96, 64], [bs, 10, 1, 224, 224]
            original_labels = label.clone()
            #! notice: Use non_blocking for better GPU transfer performance
            vid_temporal_mask_flag = vid_temporal_mask_flag.cuda(non_blocking=True)
            gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda(non_blocking=True)
            
            imgs = imgs.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask_num = 10
            label = label.view(B*mask_num, H, W)
            #! notice
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
            gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]
            
            # Optimized audio feature loading with batched processing
            audio_feature = []
            if len(audio_path) > 1:
                # Batch load for better I/O efficiency
                for audio_path_ in audio_path:
                    with open(audio_path_, 'rb') as f:
                        audio_feature1 = pickle.load(f)
                    audio_feature.append(audio_feature1)
                audio_feature = torch.cat(audio_feature, dim=0)
            else:
                # Single file load
                with open(audio_path[0], 'rb') as f:
                    audio_feature = pickle.load(f)
            
            # Move to GPU with non_blocking for better performance
            audio_feature = audio_feature.cuda(non_blocking=True)
            loss_total = 0
            lkd = 0
            l_icarl = 0
            if model_old:
                with torch.no_grad():
                    #[20,61,244,244]
                    output_old, v_map_list_old,  a_fea_list_old, head_old, attentions_old = model_old(imgs, audio_feature, vid_temporal_mask_flag)
                    #random_idx = random.randint(0, len(memory_dataset) - 1)
                    #image_old, _, _, _, _ ,_= memory_dataset[random_idx]
                    #image_old = image_old.cuda()
                    loss_FDA = 0
                    #loss_FDA = FDA_loss(imgs, image_old,audio_feature,vid_temporal_mask_flag, model, model_old)

            classif_adaptive_factor = 1.0
            mask_background = None
            mask_valid_pseudo = None
            pseudo_labels = None
            if cfg.step > 0 and cfg.method == "PLOP" and model_old:
                pseudo_labeling = cfg.pseudo
                mask_background = label < cfg.old_classes
                if pseudo_labeling == "naive":
                    label[mask_background] = output.argmax(dim=1)[mask_background]
                elif pseudo_labeling is not None and pseudo_labeling.startswith("threshold_"):
                    threshold = float(pseudo_labeling.split("_")[-1])
                    probs = torch.softmax(output_old, dim = 1)
                    pseudo_labels = probs.argmax(dim=1)
                    pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                    label[mask_background] = pseudo_labels[mask_background]
                elif pseudo_labeling == "confidence":
                    probs_old = torch.softmax(output_old, dim=1)
                    label[mask_background] = probs_old.argmax(dim=1)[mask_background]
                    sample_weights = torch.ones_like(label).to(device, dtype=torch.float32)
                    sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
                elif pseudo_labeling == "median":
                    probs = torch.softmax(output_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    pseudo_labels[max_probs < cfg.thresholds[pseudo_labels]] = 255
                    label[mask_background] = pseudo_labels[mask_background]
                elif pseudo_labeling == "entropy":
                    probs = torch.softmax(output_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    mask_valid_pseudo = (entropy(probs) /
                                         cfg.max_entropy) < cfg.thresholds[pseudo_labels]
                    if cfg.pseudo_soft is None:
                        # All old labels that are NOT confident enough to be used as pseudo labels:
                        label[~mask_valid_pseudo & mask_background] = 255

                        if cfg.pseudo_ablation is None:
                            # All old labels that are confident enough to be used as pseudo labels:
                            label[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]
                            # if global_step >=280:
                            #     print(label.min())
                            #     print(label.max())
                        elif cfg.pseudo_ablation == "corrected_errors":
                            pass  # If used jointly with data_masking=current+old, the labels already
                                  # contrain the GT, thus all potentials errors were corrected.
                        elif cfg.pseudo_ablation == "removed_errors":
                            pseudo_error_mask = label != pseudo_labels
                            kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                            removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                            label[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                            label[removed_pseudo_labels] = 255
                        else:
                            raise ValueError(f"Unknown type of pseudo_ablation={cfg.pseudo_ablation}")
                    elif cfg.pseudo_soft == "soft_uncertain":
                        label[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo & mask_background]
                    
                    if cfg.classif_adaptive_factor:
                        # Number of old/bg pixels that are certain
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                        # Number of old/bg pixels
                        den =  mask_background.float().sum(dim=(1,2))
                        # If all old/bg pixels are certain the factor is 1 (loss not changed)
                        # Else the factor is < 1, i.e. the loss is reduced to avoid
                        # giving too much importance to new pixels
                        classif_adaptive_factor = num / (den + 1e-6)
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if cfg.classif_adaptive_min_factor:
                            classif_adaptive_factor = classif_adaptive_factor.clamp(min=cfg.classif_adaptive_min_factor)
            with accelerator.autocast():
                output, v_map_list, a_fea_list, head, attentions = model(imgs, audio_feature, vid_temporal_mask_flag) # [bs*5, 24, 224, 224]
                ########################################################
                if cfg.pseudo_soft is not None:
                    loss = soft_crossentropy(
                        output,
                        label,
                        output_old,
                        mask_valid_pseudo,
                        mask_background,
                        cfg.pseudo_soft,
                        pseudo_soft_factor=cfg.pseudo_soft_factor
                    )
                elif icarl_only_dist:
                    loss = IcarlLoss(output, label, torch.sigmoid(output_old), gt_temporal_mask_flag)
                    avg_meter_icarl_loss.add({'icarl_loss': loss.item()})
                    loss_total += loss
                else:
                    #TODO： icarl_combined的时候应该是用BCEloss
                    loss, loss_dict = IouSemanticAwareLoss(output, label, a_fea_list, v_map_list, gt_temporal_mask_flag, None ,mask_background, mask_valid_pseudo, pseudo_labels,\
                                    sa_loss_flag=args.masked_av_flag, lambda_1=args.lambda_1, count_stages=args.masked_av_stages, \
                                    mask_pooling_type=args.mask_pooling_type, threshold=args.threshold_flag, norm_fea=args.norm_fea_flag, \
                                    closer_flag=args.closer_flag, euclidean_flag=args.euclidean_flag, kl_flag=args.kl_flag,cfg=cfg, classif_adaptive_factor=classif_adaptive_factor)
                    loss_total += loss
                    avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
                    avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})
                loss_total += loss_FDA
                if icarl_combined:
                    licarl = torch.nn.BCEWithLogitsLoss(reduction = "mean")
                    n_cl_old = output_old.shape[1]
                    l_icarl = cfg.icarl_importance * n_cl_old * licarl(output.narrow(1, 0, n_cl_old), torch.sigmoid(output_old))
                    loss_total += l_icarl
                    avg_meter_icarl_loss.add({'icarl_loss': l_icarl.item()})
                
                if lkd_flag and model_old:
                    # resize new output to remove new logits and keep only the old ones
                    if cfg.unkd:
                        lkd = cfg.loss_kd * UnbiasedKnowledgeDistillationLoss(output, output_old, alpha=cfg.alpha)
                    else:
                        lkd = cfg.loss_kd * KnowledgeDistillationLoss(output, output_old, alpha=cfg.alpha)
                    loss_total += lkd
                    avg_meter_lkd_loss.add({'lkd_loss': lkd.item()})
                
                #ToDO: add lde loss  怎么在不进行大改动的情况下，把features提出来, 直接加了个输出
                if lde_flag and model_old:
                    reg_importance = cfg.loss_de
                    lde_total = 0
                    lde = cfg.loss_de * lde_loss(v_map_list[0], v_map_list_old[0])
                    lde_total += lde
                    lde = cfg.loss_de * lde_loss(v_map_list[2], v_map_list_old[2])
                    lde_total += lde
                    lde = cfg.loss_de * lde_loss(v_map_list[1], v_map_list_old[1])
                    lde_total += lde
                    lde = cfg.loss_de * lde_loss(v_map_list[0], v_map_list_old[0])
                    lde_total += lde
                    loss_total += lde_total
                    avg_meter_lde_loss.add({'lde_loss': lde_total.item()})
                    
                #ToDo：修改网络结构    
                if cfg.pod is not None and cfg.step > 0:
                    attentions_old = v_map_list_old + [head_old]
                    attentions_new = v_map_list + [head]

                    # if cfg.pod_logits:
                    #     #sem_logists_small为原来的out[0]
                    #     attentions_old.append(sem_logists_small_old)
                    #     attentions_new.append(sem_logists_small)
                    if cfg.pod_large_logits:
                        attentions_old.append(output_old)
                        attentions_new.append(output)

                    pod_loss = features_distillation(
                        attentions_old,
                        attentions_new,
                        collapse_channels = cfg.pod,
                        labels=label,
                        index_new_class = cfg.old_classes,
                        pod_apply = cfg.pod_apply,
                        pod_deeplab_mask = cfg.pod_deeplab_mask,
                        pod_deeplab_mask_factor = cfg.pod_deeplab_mask_factor,
                        interpolate_last=cfg.pod_interpolate_last,
                        pod_factor=cfg.pod_factor,
                        prepro=cfg.pod_prepro,
                        deeplabmask_upscale=not cfg.deeplab_mask_downscale,
                        spp_scales=cfg.spp_scales,
                        pod_options=cfg.pod_options,
                        outputs_old=output_old,
                        use_pod_schedule=cfg.use_pod_schedule,
                        nb_current_classes=cfg.nb_current_classes,
                        nb_new_classes=cfg.nb_new_classes
                    )
                    avg_meter_pod_loss.add({'pod_loss': pod_loss.item()})
                    loss_total += pod_loss

                if cfg.entropy_min > 0. and cfg.step > 0:
                    mask_new = label > 0
                    entropies = entropy(torch.softmax(output, dim=1))
                    entropies[mask_new] = 0.
                    pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                    loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()
                    avg_meter_entropy_loss.add({'entropy_loss': loss_entmin.item()})
                    loss_total += loss_entmin
                    
            avg_meter_total_loss.add({'total_loss': loss_total.item()})
            optimizer.zero_grad()
            accelerator.backward(loss_total)
            #########################################################
            if regulizer_flag:
                if accelerator.process_index == 0:
                    regularizer.update()
                l_reg = cfg.reg_importance * regularizer.penalty()
                avg_meter_reg_loss.add({'reg_loss': l_reg})
                if l_reg !=0:
                    accelerator.backward(l_reg)
            #########################################################W
            optimizer.step()
            
            # Periodic memory cleanup to reduce CPU overhead
            if global_step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                
            global_step += 1
            if (global_step-1) % 20 == 0:
                if icarl_only_dist:
                    train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, icarl_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_icarl_loss.pop('icarl_loss'), optimizer.param_groups[0]['lr'])
                elif icarl_combined:
                        train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, icarl_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), avg_meter_icarl_loss.pop('icarl_loss'), optimizer.param_groups[0]['lr'])
                elif cfg.method == "LWF":
                        train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lkd_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'),avg_meter_lkd_loss.pop('lkd_loss'), optimizer.param_groups[0]['lr'])
                elif cfg.method == "ILT":
                        train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lde_loss:%.4f,lkd_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'),avg_meter_lde_loss.pop('lde_loss'), avg_meter_lkd_loss.pop('lkd_loss'), optimizer.param_groups[0]['lr'])
                elif cfg.method == "EWC":
                        train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, reg_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'),avg_meter_reg_loss.pop('reg_loss'), optimizer.param_groups[0]['lr'])
                elif cfg.method == "PLOP":
                        train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, pod_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_pod_loss.pop('pod_loss'),optimizer.param_groups[0]['lr'])
                else:
                    train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lr: %.6f'%(
                        global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), optimizer.param_groups[0]['lr'])
                logger.info(train_log)
        # Validation:
        if epoch >= args.start_eval_epoch and epoch % args.eval_interval == 0:
            end_task = True
            model.eval()
            metrics.reset()
            with torch.no_grad():
                for n_iter, batch_data in enumerate(val_dataloader):
                    imgs, audio_path, mask, vid_temporal_mask_flag, gt_temporal_mask_flag, _= batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 5, 1, 224, 224]

                    vid_temporal_mask_flag = vid_temporal_mask_flag.cuda(non_blocking=True)
                    gt_temporal_mask_flag  = gt_temporal_mask_flag.cuda(non_blocking=True)

                    imgs = imgs.cuda(non_blocking=True)
                    mask = mask.cuda(non_blocking=True)
                    B, frame, C, H, W = imgs.shape
                    imgs = imgs.view(B*frame, C, H, W)
                    mask = mask.view(B*frame, H, W)
                    if cfg.step > 0 and cfg.align_weight_frequency:
                        model.align_weight(cfg.align_weight)
                    elif cfg.step > 0 and cfg.align_weight_frequency == "task" and end_task:
                        model.module.align_weight(cfg.align_weight)
                    #! notice
                    vid_temporal_mask_flag = vid_temporal_mask_flag.view(B*frame) # [B*T]
                    gt_temporal_mask_flag  = gt_temporal_mask_flag.view(B*frame)  # [B*T]
                    
                    # Optimized audio feature loading for validation
                    audio_feature = []
                    if len(audio_path) > 1:
                        for audio_path_ in audio_path:
                            with open(audio_path_, 'rb') as f:
                                audio_feature1 = pickle.load(f)
                            audio_feature.append(audio_feature1)
                        audio_feature = torch.cat(audio_feature, dim=0)
                    else:
                        with open(audio_path[0], 'rb') as f:
                            audio_feature = pickle.load(f)
                    audio_feature = audio_feature.cuda(non_blocking=True)
                    output, _, _, _, _  = model(imgs, audio_feature, vid_temporal_mask_flag) # [bs*5, 21, 224, 224]
                    preds = output.detach().max(dim=1)[1]
                    targets = mask.cuda()
                    metrics.update(targets, preds)
                score = metrics.get_results()
                class_iou = list(score['Class IoU'].values())
                val_score = np.mean(class_iou[curr_idx[0]:curr_idx[1]] + [class_iou[0]])
                curr_score = np.mean( class_iou[curr_idx[0]:curr_idx[1]] )
                val_log = 'Epoch: {}, curr_score: {}, best_score: {}, class_iou: {}'.format(epoch, curr_score, best_score, class_iou)
                if curr_score > best_score:
                    if icarl_only_dist:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_icarl_best.pth')
                    elif cfg.icarl:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_icarl_combined_best.pth')
                    elif cfg.method == "LWF":
                        model_save_path = os.path.join(checkpoint_dir, 'miou_LWF_best.pth')
                    elif cfg.method == "ILT":
                        model_save_path = os.path.join(checkpoint_dir, 'miou_ILT_best.pth')
                    elif cfg.method =="EWC":
                        model_save_path = os.path.join(checkpoint_dir, 'miou_EWC_best.pth')
                    elif cfg.method == "MiB":
                        model_save_path = os.path.join(checkpoint_dir, 'miou_MiB_1_best.pth')
                    elif cfg.method == "PLOP" and cfg.shuffle:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_PLOP_shuffle_best.pth')
                    elif cfg.method == "PLOP" and cfg.memory:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_PLOP_memory_abs_best.pth')
                    elif cfg.method == "PLOP":
                        model_save_path = os.path.join(checkpoint_dir, 'miou_PLOP_best.pth')
                    elif cfg.memory:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_memory_best.pth')
                    elif args.tpavi_vv_flag:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_best_vv.pth')
                    elif args.tpavi_va_flag:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_best_va.pth')
                    if accelerator.is_main_process:
                        save_ckpt(model_save_path, model, optimizer, regularizer)
                        logger.info('save miou best model to %s'%model_save_path)               
                    best_score = curr_score
                    best_epoch = epoch
            if best_score == 0 and accelerator.is_main_process:
                if cfg.icarl:
                        model_save_path = os.path.join(checkpoint_dir, 'miou_icarl_combined_best.pth')
                        save_ckpt(model_save_path, model, optimizer, regularizer)
                        logger.info('save miou last model to %s'%model_save_path)
                else:
                    model_save_path = os.path.join(checkpoint_dir, 'miou_0_best.pth')
                    save_ckpt(model_save_path, model, optimizer, regularizer)
                    logger.info('save miou last model to %s'%model_save_path)
            model.train()
            logger.info(val_log)
        accelerator.wait_for_everyone()
    logger.info('best val score {} at peoch: {}'.format(best_score, best_epoch))

