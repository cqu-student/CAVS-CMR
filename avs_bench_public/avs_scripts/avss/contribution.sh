setting='AVSS'
# visual_backbone="resnet" # "resnet" or "pvt"
visual_backbone="resnet" # "resnet" or "pvt"


# spring.submit arun --gpu -n${gpu_num} --gres=gpu:${gpu_num}  --ntasks-per-node ${gpu_num} --quotatype=auto -p clever --job-name="train_${setting}_${visual_backbone}" \
# "
accelerate launch --gpu_ids='1' --main_process_port=26800 --num_processes 1 shapely_conrtibution.py \
    --session_name ${setting}_${visual_backbone} \
    --visual_backbone ${visual_backbone} \
    --test_batch_size 1 \
    --tpavi_stages 0 1 2 3 \

#accelerate launch --gpu_ids='4,5,6,7' --num_processes 4 --main_process_port=26700 transfer_modelold.py --max_epoches 30 --train_batch_size 2 --val_batch_size 2 --lr 0.0001 --start_eval_epoch 15 --eval_interval 2 --tpavi_stages 0 1 2 3 --tpavi_va_flag --masked_av_flag --masked_av_stages 0 1 2 3 --lambda_1 0.5 --kl_flag 

