import numpy as np
from PIL import Image
from collections import defaultdict,Counter
from color_dataloader import load_color_mask_in_PIL_to_Tensor, get_v2_pallete
from config import cfg

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
def find_different_pixel_pairs(label1_batch, label2_batch):
    """
    统计两个类别掩码中非零类别中不相同的像素值对，并合并相同对为一个三元组
    :return: 合并后的三元组列表 [(cls1, cls2, count), ...]
    """
    # 确保两个掩码的形状相同
    assert label1_batch.shape == label2_batch.shape, "两个类别掩码的形状必须相同"
    #######################################################
    mask = (label1_batch != 0) & (label2_batch != 0)
    non_equal = label1_batch != label2_batch
    non_equal = non_equal & mask
    # flattened tensor
    pixels1 = label1_batch[non_equal].tolist()
    pixels2 = label2_batch[non_equal].tolist()
    result = list(zip(pixels1,pixels2))

    count = Counter(result) #dict(tuple:int)
    if not count:
        # in case it is empty
        return 0, 0
    total = sum(count.values())
    max_pair, max_count = count.most_common()[0]
    # max_class = max_pair
    return max_pair, max_count/total