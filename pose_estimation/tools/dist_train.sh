#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

python -m pip install timm
python -m pip install einops
python -m pip install mmcv-full==1.2.2
python -m pip install numpy --upgrade
python -m pip install -r requirements.txt
pip install -e .

# COPY ZIPPED DATA
sudo apt install -y zip
mkdir /dev/shm/coco_zip
cp ../../../../dataset/coco/coco.zip /dev/shm/coco_zip
unzip /dev/shm/coco_zip/coco.zip -d /dev/shm/coco_zip

DATA_ROOT="/dev/shm/coco_zip"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [[ "$1" = @(*16xgpu*) ]]; then
    NCCL_SOCKET_IFNAME=ib0

    MASTER_IP=${MASTER_IP}
    MASTER_PORT=12345
    NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}
    PER_NODE_GPU=8 && echo PER_NODE_GPU: ${PER_NODE_GPU}
    NUM_NODE=${OMPI_COMM_WORLD_SIZE} && echo NUM_NODE: ${NUM_NODE}

    MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --nnodes=2 \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_IP \
        --master_port=$MASTER_PORT \
        $(dirname "$0")/train.py $CONFIG \
        --launcher pytorch \
        --work-dir "$(dirname $0)/../../mmpose-logs" \
        --cfg-options data_cfg.bbox_file="$DATA_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json" \
                      data.train.ann_file="$DATA_ROOT/annotations/person_keypoints_train2017.json" \
                      data.train.img_prefix="$DATA_ROOT/train2017/" \
                      data.val.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                      data.val.img_prefix="$DATA_ROOT/val2017/" \
                      data.test.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                      data.test.img_prefix="$DATA_ROOT/val2017/" 

elif [[ "$1" = @(*32xgpu*) ]]; then
    NCCL_SOCKET_IFNAME=ib0

    MASTER_IP=${MASTER_IP}
    MASTER_PORT=12345
    NODE_RANK=${OMPI_COMM_WORLD_RANK} && echo NODE_RANK: ${NODE_RANK}
    PER_NODE_GPU=8 && echo PER_NODE_GPU: ${PER_NODE_GPU}
    NUM_NODE=${OMPI_COMM_WORLD_SIZE} && echo NUM_NODE: ${NUM_NODE}

    MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
        --nproc_per_node=$GPUS \
        --nnodes=4 \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_IP \
        --master_port=$MASTER_PORT \
        $(dirname "$0")/train.py $CONFIG \
        --launcher pytorch \
        --work-dir "$(dirname $0)/../../mmpose-logs" \
        --cfg-options data_cfg.bbox_file="$DATA_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json" \
                      data.train.ann_file="$DATA_ROOT/annotations/person_keypoints_train2017.json" \
                      data.train.img_prefix="$DATA_ROOT/train2017/" \
                      data.val.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                      data.val.img_prefix="$DATA_ROOT/val2017/" \
                      data.test.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                      data.test.img_prefix="$DATA_ROOT/val2017/" 

else
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=12345 \
        $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir "$(dirname $0)/../../mmpose-logs" \
        --cfg-options data_cfg.bbox_file="$DATA_ROOT/person_detection_results/COCO_val2017_detections_AP_H_56_person.json" \
                data.train.ann_file="$DATA_ROOT/annotations/person_keypoints_train2017.json" \
                data.train.img_prefix="$DATA_ROOT/train2017/" \
                data.val.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                data.val.img_prefix="$DATA_ROOT/val2017/" \
                data.test.ann_file="$DATA_ROOT/annotations/person_keypoints_val2017.json" \
                data.test.img_prefix="$DATA_ROOT/val2017/" 

fi