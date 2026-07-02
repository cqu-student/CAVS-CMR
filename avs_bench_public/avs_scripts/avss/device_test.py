import os
import time
import nvidia_smi # pip install nvidia-ml-py3

# 设置要监控的GPU编号
gpu_ids = [4,5,6,7]

# 设置要运行的bash文件路径
bash_file = "train.sh"

# 设置显存阈值，单位为MiB
memory_threshold = 15000
nvidia_smi.nvmlInit()  # 初始化
gpu_device_count = nvidia_smi.nvmlDeviceGetCount() 
# 循环检查GPU显存是否低于阈值
while True:
    # 获取GPU显存信息
    available_gpu_id = []
    for gpu_id in gpu_ids: 
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_memory = info.free/(1024*1024) # 单位为bytes
        print("GPU {} free memory: {} MiB".format(gpu_id, free_memory))
    # 如果GPU显存低于阈值，运行bash文件并退出循环
        if free_memory > memory_threshold :
            available_gpu_id.append(gpu_id)
    if len(available_gpu_id) == len(gpu_ids):    
        os.system("sh {}".format(bash_file))
        break
    
    # 否则，等待60秒再次检查
    else:
        time.sleep(600)