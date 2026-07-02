setting='AVSS'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="pvt" # "resnet" or "pvt"


accelerate launch --multi_gpu --gpu_ids='0,1' --main_process_port=26563 --num_processes 2 test.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --test_batch_size 2 \
    --tpavi_stages 0 1 2 3 \
    --tpavi_va_flag \
    # --save_pred_mask

