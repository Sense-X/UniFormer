work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:./slowfast \
JOB_NAME=b32x4_600 
NUM_SHARDS=4
python tools/submit.py \
  --cfg $work_path/config.yaml \
  --job_dir $work_path/${JOB_NAME}/  \
  --num_shards $NUM_SHARDS \
  --num_gpus 8 \
  --partition your_partition \
  --name ${JOB_NAME} \
  DATA.PATH_TO_DATA_DIR ./data_list/k600 \
  DATA.PATH_PREFIX your_path_to_data \
  DATA.PATH_LABEL_SEPARATOR "," \
  MODEL.NUM_CLASSES 600 \
  DATA_LOADER.NUM_WORKERS 8 \
  TRAIN.EVAL_PERIOD 5 \
  TRAIN.CHECKPOINT_PERIOD 1 \
  TRAIN.BATCH_SIZE 32 \
  TEST.BATCH_SIZE 32 \
  NUM_SHARDS $NUM_SHARDS \
  NUM_GPUS 8 \
  UNIFORMER.DROP_DEPTH_RATE 0.3 \
  SOLVER.MAX_EPOCH 110 \
  SOLVER.BASE_LR 1e-4 \
  SOLVER.WARMUP_EPOCHS 10.0 \
  DATA.TEST_CROP_SIZE 224 \
  TEST.NUM_ENSEMBLE_VIEWS 1 \
  TEST.NUM_SPATIAL_CROPS 1 \
  RNG_SEED 6666 \
  OUTPUT_DIR $work_path