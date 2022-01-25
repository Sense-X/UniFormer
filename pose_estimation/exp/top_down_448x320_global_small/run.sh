#!/usr/bin/env bash

work_path=$(dirname $0)
PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH GLOG_vmodule=MemcachedClient=-1 \

srun -p GVT -n1 --gres=gpu:8 --ntasks-per-node=1 --cpus-per-task=48 --quotatype=reserved --job-name lkc_s \
python -m torch.distributed.launch --nproc_per_node=8 \
    tools/train.py ${work_path}/config.py \
    --launcher pytorch \
    --cfg-options model.backbone.pretrained_path='/mnt/lustre/cuiyufeng/likunchang/model/uniformer/uniformer_small_in1k.pth' \
    --work-dir ${work_path}/ckpt \
    --resume-from ${work_path}/ckpt/config/latest.pth \
    2>&1 | tee -a ${work_path}/log.txt
