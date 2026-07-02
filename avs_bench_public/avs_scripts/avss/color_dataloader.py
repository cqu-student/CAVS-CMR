import os
# from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
import json

# import cv2
from PIL import Image
from torchvision import transforms
import sys
from config import cfg
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(root_dir, 'tasks'))
import torchvision
import get_tasks


def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0) :
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls) # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete


def crop_resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img

def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img

def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label


def load_image_in_PIL_to_Tensor(path, split='train', mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if cfg.DATA.CROP_IMG_AND_MASK:
        if split == 'train' or split == "memory":
            img_PIL = crop_resize_img(cfg.DATA.CROP_SIZE, img_PIL, img_is_mask=False)
        else:
            img_PIL = resize_img(cfg.DATA.CROP_SIZE, img_PIL, img_is_mask=False)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB'):
    color_mask_PIL = Image.open(path).convert(mode)
    if cfg.DATA.CROP_IMG_AND_MASK:
        if split == 'train' or split == "memory":
            color_mask_PIL = crop_resize_img(cfg.DATA.CROP_SIZE, color_mask_PIL, img_is_mask=True)
        else:
            color_mask_PIL = resize_img(cfg.DATA.CROP_SIZE, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label) # [H, W]
    color_label = color_label.unsqueeze(0)
    return color_label # both [1, H, W]
    

def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel


class V2Dataset(Dataset):
    """Dataset for audio visual semantic segmentation of AVSBench-semantic (V2)"""
    def __init__(self, split='train', debug_flag=False):
        super(V2Dataset, self).__init__()
        self.split = split
        #分类类别数量
        self.mask_num = cfg.MASK_NUM
        #读取所有的训练数据
        df_all = pd.read_csv(cfg.DATA.META_CSV_PATH, sep=',')
        self.length = len(df_all)
        #split用于划分训练集和测试集
        if self.split == 'memory':
            df_split = pd.read_csv(cfg.DATA.MEMORY_CSV_PATH, sep=',')
        elif self.split != 'test' and self.split !='memory':
            df_split = df_all[df_all['split'] == split]
            label_index, label_old = get_tasks.get_task_labels(cfg.dataset_name, cfg.task_name, cfg.step)
            with open('../../avsbench_data/AVSBench_semantic/idx2label.json', 'r') as fr:
                index_to_label = json.load(fr)
            label_array = [index_to_label[str(index+1)] for index in label_index]
            df_split.loc[:,'a_obj_split'] = df_split['a_obj'].str.split('_')
            df_split = df_split[df_split['a_obj_split'].apply(lambda x: any(label in label_array for label in x))]
            
        else:
            df_split = df_all[df_all['split'] == split]
        self.df_split = df_split
        print("{}/{} videos are used for {}.".format(len(df_split), len(df_all), self.split))
        if debug_flag:
            self.df_split = self.df_split[:100]
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.v2_pallete = get_v2_pallete(cfg.DATA.LABEL_IDX_PATH, num_cls=cfg.NUM_CLASSES)
    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, set = df_one_video['uid'], df_one_video['label']
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
            img = load_image_in_PIL_to_Tensor(img_path, split=self.split, transform=self.img_transform)
            imgs.append(img)
        for pad_i in range(imgs_pad_zero_num): #! pad black image?
            img = torch.zeros_like(img)
            imgs.append(img)

        labels = []
        mask_path_list = sorted(os.listdir(color_mask_base_path))
        for mask_path in mask_path_list:
            if not mask_path.endswith(".png"):
                mask_path_list.remove(mask_path)
        mask_num = len(mask_path_list)
        if self.split != 'train' and self.split != "memory":
            if set == 'v2':
                assert mask_num == 10
            else:
                assert mask_num == 5

        #get the index order and the transform
        label_index, label_old = get_tasks.get_task_labels(cfg.dataset_name, cfg.task_name, cfg.step)
        label_old = label_old if label_old is not None else []
        order = label_old + label_index
        if not (0 in order):
            order = [0] + order
        masking_value = 0 if self.split == 'train' else 255
        if self.split == "test" or self.split == "memory":
            masking = False
        else:
            masking = True
        inverted_order = {label: order.index(label) for label in order}
        inverted_order[255] = masking_value
        reorder_transform = torchvision.transforms.Lambda(
                lambda t: t.apply_(lambda x: inverted_order[x] if x in inverted_order else masking_value))
        #!notice: 测试
        if masking:
            #这个目的是为了建立当前应该存在的类别
            if 0 in label_index:
                tmp_labels = label_index + [255]
            else:
                tmp_labels = [0] + label_index + [255]
            target_transform = torchvision.transforms.Lambda(
                lambda t: t.apply_(lambda x: inverted_order[x] if x in tmp_labels else masking_value))
        #prevent the class from the future
        else:
            target_transform = reorder_transform

        mask_num = len(mask_path_list)
        label_pad_zero_num = 10 - mask_num
        for mask_id in range(mask_num):
            mask_path = os.path.join(color_mask_base_path, "%d.png"%(mask_id))
            color_label = load_color_mask_in_PIL_to_Tensor(mask_path, v_pallete=self.v2_pallete, split=self.split)
            color_label = target_transform(color_label)
            labels.append(color_label)
        #没有图像的添0标签
        for pad_j in range(label_pad_zero_num):
            color_label = torch.zeros_like(color_label)
            labels.append(color_label)

        imgs_tensor = torch.stack(imgs, dim=0)
        labels_tensor = torch.stack(labels, dim=0)

        return imgs_tensor, audio_path, labels_tensor, \
             vid_temporal_mask_flag, gt_temporal_mask_flag, video_name

    def __len__(self):
        return len(self.df_split)

    @property
    def num_classes(self):
        """Number of categories (including background)."""
        return cfg.NUM_CLASSES

    @property
    def classes(self):
        """Category names."""
        with open(cfg.DATA.LABEL_IDX_PATH, 'r') as fr:
            classes = json.load(fr)
        return [label for label in classes.keys()]