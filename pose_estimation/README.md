# COCO pose estimation with UniFormer

We currenent release the code and models for:

- [x] Top-down

  


## Updates

**01/18/2022**

**\[Initial commits\]:** 

1. Models with Top-down.

   


## Model Zoo

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/14iH-q9BXSY3xGtIzzfudUsnKhoRf000h?usp=sharing), [total_logs](https://drive.google.com/drive/folders/14pmCt-dY_WrkHLxO66hsCuH2bSShs7Jm?usp=sharing). 
>
> We also release the models on **Baidu Cloud**: [total_models (lqht)](https://pan.baidu.com/s/1XqeHNNaxPGdb6BncG4TDCA), [total_logs (j7e2)](https://pan.baidu.com/s/1e-zS3auB_Y7ksCQdUAS6nA).

### Note

- All the models are pretrained on ImageNet-1K without [Token Labeling](https://arxiv.org/abs/2104.10858) and [Layer Scale](https://arxiv.org/abs/2103.17239). Reason can be found in [issue \#12](https://github.com/Sense-X/UniFormer/issues/12#issuecomment-1044001497).

### Top-Down

|   Backbone    | Input Size | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR<sup>M</sup> | AR<sup>L</sup> | AR | FLOPs |                            Model                             |                             Log                              |                            Shell                             |
| :----------- | :-------- | :------------- | :-- | :--- | :---------------------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------- | :------------- | :------------- | :------------- | :------------- |
|  UniFormer-S  | 256x192 | 74.0 | 90.3 | 82.2 | 66.8 | 76.7 | 79.5 | 4.7G | [google](https://drive.google.com/file/d/162R0JuTpf3gpLe1IK6oxRoQK7JSj4ylx/view?usp=sharing) | [google](https://drive.google.com/file/d/15j40u97Db6TA2gMHdn0yFEsDFb5SMBy4/view?usp=sharing) | [run.sh](exp/top_down_256x192_global_small/run.sh)/[config](exp/top_down_256x192_global_small/config.py) |
|  UniFormer-S  | 384x288 | 75.9 | 90.6 | 83.4 | 68.6 | 79.0 | 81.4 | 11.1G | [google](https://drive.google.com/file/d/163vuFkpcgVOthC05jCwjGzo78Nr0eikW/view?usp=sharing) | [google](https://drive.google.com/file/d/15X9M_5cq9RQMgs64Yn9YvV5k5f0zOBHo/view?usp=sharing) | [run.sh](exp/top_down_384x288_global_small/run.sh)/[config](exp/top_down_384x288_global_small/config.py) |
|  UniFormer-S  | 448x320 | 76.2 | 90.6 | 83.2 | 68.6 | 79.4 | 81.4 | 14.8G | [google](https://drive.google.com/file/d/165nQRsT58SXJegcttksHwDn46Fme5dGX/view?usp=sharing) | [google](https://drive.google.com/file/d/15IJjSWp4R5OybMdV2CZEUx_TwXdTMOee/view?usp=sharing) | [run.sh](exp/top_down_448x320_global_small/run.sh)/[config](exp/top_down_448x320_global_small/config.py) |
|  UniFormer-B  | 256x192 | 75.0 | 90.6 | 83.0 | 67.8 | 77.7 | 80.4 | 9.2G | [google](https://drive.google.com/file/d/15tzJaRyEzyWp2mQhpjDbBzuGoyCaJJ-2/view?usp=sharing) | [google](https://drive.google.com/file/d/15jJyTPcJKj_id0PNdytloqt7yjH2M8UR/view?usp=sharing) | [run.sh](exp/top_down_256x192_global_base/run.sh)/[config](exp/top_down_256x192_global_base/config.py) |
| UniFormer-B | 384x288 | 76.7 | 90.8 | 84.0 | 69.3 | 79.7 | 81.4 | 14.8G | [google](https://drive.google.com/file/d/15qtUaOR_C7-vooheJE75mhA9oJQt3gSx/view?usp=sharing) | [google](https://drive.google.com/file/d/15L1Uxo_uRSMlGnOvWzAzkJLKX6Qh_xNw/view?usp=sharing) | [run.sh](exp/top_down_384x288_global_base/run.sh)/[config](exp/top_down_384x288_global_base/config.py) |
| UniFormer-B | 448x320 | 77.4 | 91.1 | 84.4 | 70.2 | 80.6 | 82.5 | 29.6G | [google](https://drive.google.com/file/d/156iNxetiCk8JJz41aFDmFh9cQbCaMk3D/view?usp=sharing) | [google](https://drive.google.com/file/d/15aRpZc2Tie5gsn3_l-aXto1MrC9wyzMC/view?usp=sharing) | [run.sh](exp/top_down_448x320_global_base/run.sh)/[config](exp/top_down_448x320_global_base/config.py) |

## Usage

### Installation

Please refer to [get_started](https://github.com/open-mmlab/mmpose/blob/master/docs/en/getting_started.md) for installation and dataset preparation.

### Training

1. Download the pretrained models in our repository.

2. Simply run the training scripts in [exp](exp) as followed:

   ```shell
   bash ./exp/top_down_256x192_global_small/run.sh
   ```

   Or you can train other models as follower:

   ```shell
   # single-gpu training
   python tools/train.py <CONFIG_FILE> --cfg-options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments]
   
   # multi-gpu training
   tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments] 
   ```

**[Note]:**

- We use global MHRA during training and set the corresponding hyperparameters in the `config.py`:

  ```python
  window: False, # whether use window MHRA
  hybrid: False, # whether use hybrid MHRA
  ```
  
- To avoid **out of memory**, we can use `torch.utils.checkpoint`  in the `config.py`:

  ```python
  use_checkpoint=True, # whether use checkpoint
  checkpoint_num=[0, 0, 2, 0], # index for using checkpoint in every stage
  ```

### Testing

```shell
# single-gpu testing
python tools/test.py <CONFIG_FILE> <POSE_CHECKPOINT_FILE> --eval mAP

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <POSE_CHECKPOINT_FILE> <GPU_NUM> --eval mAP
```

## Acknowledgement

This repository is built based on [mmpose](https://github.com/open-mmlab/mmpose) and [HRT](https://github.com/HRNet/HRFormer/tree/main/pose) repository.

