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
memory_csv = '../../avsbench_data/AVSBench_semantic/memory_abs_60_5_v2s.csv'
# memory_csv = f'../../avsbench_data/AVSBench_semantic/memory.csv'
N_CLASSES = sum(get_tasks.get_per_task_classes(cfg.dataset_name,cfg.task_name,cfg.step))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_all = pd.read_csv('../../avsbench_data/AVSBench_semantic/memory_all_60_5_v2s.csv', sep=',')
df_split = df_all[df_all['split'] == 'val']
label_index, _ = get_tasks.get_task_labels(cfg.dataset_name, cfg.task_name, cfg.step)
print(len(label_index))  
with open('../../avsbench_data/AVSBench_semantic/idx2label.json', 'r') as fr:
    index_to_label = json.load(fr)
label_array = [index_to_label[str(index+1)] for index in label_index]
df_split.loc[:,'a_obj_split'] = df_split['a_obj'].str.split('_')
df_split = df_split[df_split['a_obj_split'].apply(lambda x: any(label in label_array for label in x))]
df_split = df_all[df_all['split'] == 'train']
label_index, _ = get_tasks.get_task_labels(cfg.dataset_name, cfg.task_name, cfg.step)
print(len(label_index))  
with open('../../avsbench_data/AVSBench_semantic/idx2label.json', 'r') as fr:
    index_to_label = json.load(fr)
label_array = [index_to_label[str(index+1)] for index in label_index]
df_split.loc[:,'a_obj_split'] = df_split['a_obj'].str.split('_')
df_split = df_split[df_split['a_obj_split'].apply(lambda x: any(label in label_array for label in x))]
if os.path.exists(memory_csv):
    result_df = pd.read_csv(memory_csv, sep=',')
else:
    result_df = pd.DataFrame()
for label in label_array:
    if label == "background":
        continue
    filtered_rows = df_split[df_split['a_obj_split'].apply(lambda x: label in x)]
    # 根据 contribution 列排序，取最高的 5 行
    filtered_rows['abs_diff'] = filtered_rows['contribution'].abs()
    # top_rows = filtered_rows.sample(n=3, random_state=1)
    # top_rows = filtered_rows.nlargest(5, 'contribution')
    top_rows = filtered_rows.nsmallest(5, 'abs_diff')
    # top_rows = pd.concat([top_rows1, top_rows2])
    # 如果不足 5 行，则全选
    if len(top_rows) < 5:
        top_rows = filtered_rows
    print(top_rows)
        # 将筛选出的行添加到 result_df 中
    result_df = pd.concat([result_df, top_rows])
result_df.to_csv(memory_csv, index=False)