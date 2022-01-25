#!/usr/bin/env bash

work_path='./exp/mask_rcnn_uniformer_s_hybrid_patch4_window7_mstrain_480-800_adamw_dp0.1_3x_coco'
GLOG_vmodule=MemcachedClient=-1 \

srun -p DBX32 -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=48 --quotatype=spot \
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train.py ${work_path}/config.py \
    --launcher pytorch \
    --cfg-options model.backbone.pretrained_path='/mnt/lustre/zongzhuofan/others/lkc/model/uniformer/uniformer_small_in1k.pth' \
    --work-dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt