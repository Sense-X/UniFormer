work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python3 validate.py your_path_to_data \
  --model uniformer_large_ls \
  --checkpoint your_path_to_checkpoint \
  --no-test-pool \
  --img-size 384 \
  --batch-size 32 \
  2>&1 | tee ${work_path}/log.txt