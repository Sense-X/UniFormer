# COCO Object detection with UniFormer

We currenent release the code and models for:

- [x] Mask R-CNN

- [x] Cascade Mask R-CNN

  

## Updates


**05/22/2023**

Lightweight models with Mask R-CNN are released.

**01/18/2022**

1. Models with Mask R-CNN are released.

2. Models with Cascade Mask R-CNN are released.

   

## Model Zoo

>  The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/12tOzgxF0e4fdqOHGOQFlM5eakrlKx3Ml?usp=sharing), [total_logs](https://drive.google.com/drive/folders/12wFRaILxeITdCNYZj1S38n4iaNn9trt8?usp=sharing). 
>
> We also release the models on **Baidu Cloud**: [total_models (5v6i)](https://pan.baidu.com/s/16hSrbdPxodHvcx6GcSKGsw), [total_logs (wr74)](https://pan.baidu.com/s/1RFM4X7NgjtCM-1OlWiGdyA).

### Note

- All the models are pretrained on ImageNet-1K without [Token Labeling](https://arxiv.org/abs/2104.10858) and [Layer Scale](https://arxiv.org/abs/2103.17239). Reason can be found in [issue \#12](https://github.com/Sense-X/UniFormer/issues/12#issuecomment-1044001497).
- The FLOPs are measured at resolution 800×1280.

### Mask R-CNN

| Backbone | Lr Schd | box mAP | mask mAP | #params | FLOPs | Model | Log | Shell |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |:--- |
| UniFormer-XXS | 1x | 42.8 | 39.2 | 29.4M | - | [google](https://drive.google.com/file/d/1CUyS-hWbpLvOXHZ35n4BZPmiCbMXt0mc/view?usp=sharing) | [google](https://drive.google.com/file/d/1CDw5Ql409xczOI8M9kA4CNJCgemr__U5/view?usp=sharing) | [run.sh](./exp_light/mask_rcnn_1x_xxs/run.sh)/[config](./exp_light/mask_rcnn_1x_xxs/config.py) |
| UniFormer-XS  | 1x | 44.6 | 40.9 | 35.6M | - | [google](https://drive.google.com/file/d/1CRHb-W5Csy_509J4Ux62TOT7dpixdsOi/view?usp=sharing) | [google](https://drive.google.com/file/d/1C90FNxJ4GV2QFLRtIWHYG3CGuA8Rw6g-/view?usp=sharing) | [run.sh](./exp_light/mask_rcnn_1x_xs/run.sh)/[config](./exp_light/mask_rcnn_1x_xs/config.py) |
| UniFormer-S<sub>h14</sub> | 1x | 45.6 | 41.6 | 41M | 269G | [google](https://drive.google.com/file/d/13KhBYkHKQg-CyhAgn1LQM1K0R4bwSpWT/view?usp=sharing) | [google](https://drive.google.com/file/d/130ys7x9VJ_F0qwBMZSdjIXpN_nY2V3yT/view?usp=sharing) | [run.sh](exp/mask_rcnn_1x_hybrid_small/run.sh)/[config](exp/mask_rcnn_1x_hybrid_small/config.py) |
| UniFormer-S<sub>h14</sub> | 3x+MS | 48.2 | 43.4 | 41M | 269G | [google](https://drive.google.com/file/d/13O6P-spXbWuXgUjLw8AyRITYTwhdOwuD/view?usp=sharing) | [google](https://drive.google.com/file/d/1337Ee1HWpUl6L9NupO2XjUfkcUrfD2-P/view?usp=sharing) | [run.sh](exp/mask_rcnn_3x_ms_hybrid_small/run.sh)/[config](exp/mask_rcnn_3x_ms_hybrid_small/config.py) |
| UniFormer-B<sub>h14</sub> | 1x | 47.4 | 43.1 | 69M | 399G | [google](https://drive.google.com/file/d/13YrVEiqFraSuX1NzdhPnhw7bFXTvEii1/view?usp=sharing) | [google](https://drive.google.com/file/d/12yIcaEs9sHoDNwrKOvWTCV2oP0JO31-I/view?usp=sharing) | [run.sh](exp/mask_rcnn_1x_hybrid_base/run.sh)/[config](exp/mask_rcnn_1x_hybrid_base/config.py) |
| UniFormer-B<sub>h14</sub> | 3x+MS | 50.3 | 44.8 | 69M | 399G | [google](https://drive.google.com/file/d/12xyS2xqFSYMkHUiVYVGVW7gaLIxuK2q6/view?usp=sharing) | [google](https://drive.google.com/file/d/13-HUaDv3EmfHqTuatMaoDkay0myO5xPY/view?usp=sharing) | [run.sh](exp/mask_rcnn_3x_ms_hybrid_base/run.sh)/[config](exp/mask_rcnn_3x_ms_hybrid_base/config.py) |

### Cascade Mask R-CNN

| Backbone | Lr Schd | box mAP | mask mAP | #params | FLOPs | Model | Log | Shell |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |:--- |
| UniFormer-S<sub>h14</sub> | 3x+MS | 52.1 | 45.2 | 79M | 747G | [google](https://drive.google.com/file/d/13IhpRkh2hE8WfDyw-5-NXSUVZjn9DX20/view?usp=sharing) | [google](https://drive.google.com/file/d/12zBocGyAOsKdojyleJFuZ1UbuU5kmURA/view?usp=sharing) | [run.sh](exp/cascade_mask_rcnn_3x_ms_hybrid_small/run.sh)/[config](exp/cascade_mask_rcnn_3x_ms_hybrid_small/config.py) |
| UniFormer-B<sub>h14</sub> | 3x+MS | 53.8 | 46.4 | 107M | 878G | [google](https://drive.google.com/file/d/13G9wc73CmS1Kb-kVelFSDlK-ezUBadzQ/view?usp=sharing) | [google](https://drive.google.com/file/d/1360aWgKvE29rmcTi__cqAtPtPL19QZUu/view?usp=sharing) | [run.sh](exp/cascade_mask_rcnn_3x_ms_hybrid_base/run.sh)/[config](exp/cascade_mask_rcnn_3x_ms_hybrid_base/config.py) |

## Usage

### Installation

Please refer to [get_started](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Training

1. Download the pretrained models in our repository.

2. Simply run the training scripts in [exp](exp) as followed:

   ```shell
   bash ./exp/mask_rcnn_1x_hybrid_small/run.sh
   ```

   Or you can train other models as follower:

   ```shell
   # single-gpu training
   python tools/train.py <CONFIG_FILE> --cfg-options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments]
   
   # multi-gpu training
   tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments] 
   ```

**[Note]:**

- We use hybrid MHRA to reduce training cost and set the corresponding hyperparameters in the `config.py`:

  ```python
  window: False, # whether use window MHRA
  hybrid: True, # whether use hybrid MHRA
  window_size: 14, # size of window (>=14)
  ```

- To avoid **out of memory**, we use `torch.utils.checkpoint`  in the `config.py`:

  ```python
  use_checkpoint=True, # whether use checkpoint
  checkpoint_num=[0, 0, 8, 0], # index for using checkpoint in every stage
  ```

### Testing

```shell
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

## Acknowledgement

This repository is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) repository.

