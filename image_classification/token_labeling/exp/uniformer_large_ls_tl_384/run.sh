work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python3 -m torch.distributed.launch --nproc_per_node=8 main.py your_path_to_data \
  --model uniformer_small_plus \
  --epochs 50 \
  --warmup-epochs 0 \
  --batch-size 32 \
  --apex-amp \
  --img-size 384 \
  --drop-path 0.3 \
  --token-label \
  --token-label-data your_path_to_label_data \
  --token-label-size 12 \
  --lr 5.e-6 \
  --min-lr 5.e-6 \
  --weight-decay 1.e-8 \
  --finetune your_path_to_checkpoint \
  --output ${work_path} \
  2>&1 | tee ${work_path}/log.txt
