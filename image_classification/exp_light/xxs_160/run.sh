work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_light.py \
    --data-path your_path_to_data \
    --model uniformer_xxs \
    --input-size 160 \
    --batch-size 512 \
    --lr 4e-4 \
    --model-ema-decay 0.9999 \
    --drop-path 0 \
    --no-repeated-aug \
    --aa v0 \
    --weight-decay 0.1 \
    --warmup-epochs 30 \
    --epoch 600 \
    --dist-eval \
    --output_dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt
