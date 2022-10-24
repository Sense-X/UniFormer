work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python3 validate.py your_path_to_data \
  --model uniformer_small_dim64 \
  --checkpoint your_path_to_checkpoint \
  --no-test-pool \
  --img-size 224 \
  --batch-size 128 \
  2>&1 | tee ${work_path}/log.txt
