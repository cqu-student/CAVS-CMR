import numpy as np
import torch
import sys
import os
from torchvggish import vggish
from config import cfg
import pickle
import pdb
data_root = "../../avsbench_data/AVSBench_semantic"
subdirectories = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
subdirectories = subdirectories[1:]
subdirectories = ['v1s']
class audio_extractor(torch.nn.Module):
    def __init__(self, cfg, device):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(cfg, device)

    def forward(self, audio):
        audio_fea = self.audio_backbone(audio)
        return audio_fea
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_backbone = audio_extractor(cfg, device)
for index in subdirectories:
    if index == "v1s":
        vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
        gt_temporal_mask_flag  = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])#.bool()
    elif index == "v1m":
        vid_temporal_mask_flag = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
        gt_temporal_mask_flag  = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])#.bool()
    elif index == "v2":
        vid_temporal_mask_flag = torch.ones(10)#.bool()
        gt_temporal_mask_flag = torch.ones(10)#.bool()
    vid_temporal_mask_flag = vid_temporal_mask_flag.to(device)
    files = os.listdir(os.path.join(data_root, index))
    for file in files:
        audio_path = os.path.join(data_root, index, file,"audio.wav")
        audio1_path = os.path.join(data_root, index, file,"audio1.pkl")
        with torch.no_grad():
            print(audio_path)
            audio_feature = audio_backbone([audio_path])
            audio_feature = audio_feature * vid_temporal_mask_flag.unsqueeze(-1)
            with open(audio1_path, 'wb') as f:
                pickle.dump(audio_feature, f)
            
