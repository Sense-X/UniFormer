work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --data-path your_path_to_data \
    --model uniformer_small \
    --batch-size 128 \
    --drop-path 0.1 \
    --epoch 300 \
    --dist-eval \
    --output_dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt
