import math
import json
import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
import pandas as pd
from color_dataloader import V2Dataset
import get_tasks
import copy

def memory_sampling(samples, df_memory):
    index_1 = []
    with open('../../avsbench_data/AVSBench_semantic/idx2label.json', 'r') as fr:
        index_to_label = json.load(fr)
    label_array = [index_to_label[str(sample+1)] for sample in samples]
    for sample in label_array:
        df_split = copy.deepcopy(df_memory)
        df_split.loc[:,'a_obj_split'] = df_split['a_obj'].str.split('_')
        df_split = df_split[df_split['a_obj_split'].apply(lambda x: sample in x)]
        random_index = np.random.choice(df_split.index)
        index_1.append(random_index)
    return index_1

#最后只需要保存文件名
def memory_sampling_balanced(opts,args,old_model):
    train_dataset = V2Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        persistent_workers=True)
    
    num_classes = get_tasks.get_per_task_classes(opts.dataset_name, opts.task_name, opts.step)
    prev_num_classes = sum(num_classes[:-1])
    df_all = pd.read_csv(opts.DATA.META_CSV_PATH, sep=',') #读进所有的数据，后面要照着查找
    memory_csv = f'../../avsbench_data/AVSBench_semantic/memory.csv'
    if opts.step > 1:
        df_memory = pd.read_csv(memory_csv)
        df_memory_candidate = df_memory[df_memory['step'] == opts.step-1] #回放上一个step的memory
    else:
        memory_list = []
        memory_candidates = []
    
    print("...start memory candidates collection")
    import pickle
    for imgs, audio_path, label, vid_temporal_mask_flag, gt_temporal_mask_flag, video_name in train_dataloader:
        with torch.no_grad():
            if opts.step > 1:
                imgs = imgs.cuda()
                label = label.cuda()
                audio_feature = []
                for audio_path_ in audio_path:
                    with open(audio_path_, 'rb') as f:
                        audio_feature1 = pickle.load(f)
                    audio_feature.append(audio_feature1)
                audio_feature = torch.cat(audio_feature, dim=0).cuda()
                outputs_old  = old_model(imgs, audio_feature, vid_temporal_mask_flag)
                pred_logits = torch.softmax(outputs_old, dim=1)
                pred_scores, pred_labels = torch.max(pred_logits, dim=1)
                '''pseudo labeling'''
                label = torch.where((label <= 0) & (pred_labels > 0) & (pred_scores >= 0.7), pred_labels, label)
        
        for b in range(imgs.size(0)):
            img_name = video_name[b]
            target = label[b]

            labels = torch.unique(target).detach().cpu().numpy()
            labels = labels.tolist()
            if -1 in labels:
                labels.remove(-1)
            if 0 in labels:
                labels.remove(0)
            
            objs_num = len(labels)
            objs_ratio = int((target > 0).sum())

            memory_candidates.append([img_name, objs_num, objs_ratio, labels])

        print("...end memory candidates collection : ", len(memory_candidates))

        #####################################################################################

        print("...start memory list generation")
        curr_memory_list = {f"class_{cls}":[] for cls in range(1, prev_num_classes)}
        sorted_memory_candidates = memory_candidates.copy()
        np.random.shuffle(sorted_memory_candidates)
        random_class_order = list(range(1, prev_num_classes))
        np.random.shuffle(random_class_order)
        num_sampled = 0



        # while opts.mem_size > num_sampled:
        if opts.mem_size > num_sampled:
            for cls in random_class_order:
                for idx, mem in enumerate(sorted_memory_candidates):
                    img_name, objs_num, objs_ratio, labels = mem

                    if cls in labels:
                        curr_memory_list[f"class_{cls}"].append(mem)
                        num_sampled += 1
                        del sorted_memory_candidates[idx]
                        # break
                    
                if opts.mem_size <= num_sampled:
                    break
    ###################################### 
    '''save memory info'''
    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    sorted_list = pd.DataFrame()
    for mem in sampled_memory_list:
        video_name = mem[0]
        matched_rows = df_all[df_all['uid']==video_name]
        sorted_list = pd.concat([sorted_list, matched_rows], ignore_index=True)
    with open(memory_csv,"w") as f:
        f.truncate()
    sorted_list.to_csv(memory_csv, index=False)

    # """ save memory info """
    # sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    # memory_list[f"step_{opts.step}"] = {"memory_candidates": sampled_memory_list, 
    #                                               "memory_list": sorted([mem[0] for mem in sampled_memory_list])
    #                                           }
    # with open(memory_csv, "w") as json_file:
    #     json.dump(memory_list, json_file)




    

