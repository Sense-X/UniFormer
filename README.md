# UniFormer

This repo is the official implementation of  ["Uniformer: Unified Transformer for Efficient Spatiotemporal Representation Learning"](). It currently includes code and models for the following tasks:

- [x]  [Imgae Classification](image_classification)

- [x]  [Video Classification](video_classification)

- [ ]  <font color= #A6ACAF>Object Detection (code will be released soon)</font>

- [ ]  <font color= #A6ACAF>Semantic Segmentation (code will be released soon)</font>

- [ ]  <font color= #A6ACAF>Pose Estimation (code will be released soon)</font>

  

## Updates

***01/13/2022***

**\[Initial commits\]:**

1. Pretrained models on ImageNet-1K, Kinetics-400, Kinetics-600, Something-Something V1&V2

2. The supported code and models for image classification and video classification are provided.

   

## Introduction

**UniFormer** (**Uni**fied trans**Former**) is introduce in [arxiv](), which effectively unifies 3D convolution and spatiotemporal self-attention in a concise transformer format. We adopt local MHRA in shallow layers to largely reduce computation burden and global MHRA in deep layers to learn global token relation. 

UniFormer achieves strong performance on video classification. With only ImageNet-1K pretraining,  our UniFormer achieves **82.9%/84.8%** top-1 accuracy on Kinetics-400/Kinetics-600, while requiring **10x** fewer GFLOPs than other comparable methods (e.g., 16.7x fewer GFLOPs than [ViViT](https://openaccess.thecvf.com/content/ICCV2021/papers/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.pdf) with JFT-300M pre-training). For Something-Something V1 and V2, our UniFormer achieves **60.9%** and **71.2%** top-1 accuracy respectively, which are new state-of-the-art performances. 

![teaser](figures/framework.png)

## Main results on ImageNet-1K

Please see [image_classification](image_classification)  for more details.

More models with large resolution and token labeling will be released  soon.

| Model        | Pretrain    | Resolution | Top-1 | #Param. | FLOPs |
| ------------ | ----------- | ---------- | ----- | ------- | ----- |
| UniFormer-S  | ImageNet-1K | 224x224    | 82.9  | 22M     | 3.6G  |
| UniFormer-S† | ImageNet-1K | 224x224    | 83.4  | 24M     | 4.2G  |
| UniFormer-B  | ImageNet-1K | 224x224    | 83.9  | 50M     | 8.3G  |

## Main results on Kinetics-400

Please see [video_classification](video_classification)  for more details.

| Model       | Pretrain    | #Frame | Sampling Method | FLOPs | K400 Top-1 | K600 Top-1 |
| ----------- | ----------- | ------ | --------------- | ----- | ---------- | ---------- |
| UniFormer-S | ImageNet-1K | 16x1x4 | 16x4            | 167G  | 80.8       | 82.8       |
| UniFormer-S | ImageNet-1K | 16x1x4 | 16x8            | 167G  | 80.8       | 82.7       |
| UniFormer-S | ImageNet-1K | 32x1x4 | 32x4            | 438G  | 82.0       | -          |
| UniFormer-B | ImageNet-1K | 16x1x4 | 16x4            | 387G  | 82.0       | 84.0       |
| UniFormer-B | ImageNet-1K | 16x1x4 | 16x8            | 387G  | 81.7       | 83.4       |
| UniFormer-B | ImageNet-1K | 32x1x4 | 32x4            | 1036G | 82.9       | 84.5*      |

\* Since Kinetics-600 is too large to train (>1 month in single node with 8 A100 GPUs), we provide model trained in multi node (around 2 weeks with 32 V100 GPUs), but the result is lower due to the lack of tuning hyperparameters.

## Main results on Something-Something

Please see [video_classification](video_classification)  for more details.

| Model       | Pretrain | #Frame | FLOPs | SSV1 Top-1 | SSV2 Top-1 |
| ----------- | -------- | ------ | ----- | ---------- | ---------- |
| UniFormer-S | K400     | 16x3x1 | 125G  | 57.2       | 67.7       |
| UniFormer-S | K600     | 16x3x1 | 125G  | 57.6       | 69.4       |
| UniFormer-S | K400     | 32x3x1 | 329G  | 58.8       | 69.0       |
| UniFormer-S | K600     | 32x3x1 | 329G  | 59.9       | 70.4       |
| UniFormer-B | K400     | 16x3x1 | 290G  | 59.1       | 70.4       |
| UniFormer-B | K600     | 16x3x1 | 290G  | 58.8       | 70.2       |
| UniFormer-B | K400     | 32x3x1 | 777G  | 60.9       | 71.1       |
| UniFormer-B | K600     | 32x3x1 | 777G  | 61.0       | 71.2       |

## Main results on downstream tasks

We have conducted extensive experiments on downstream tasks and achieved comparable results with SOTA models.

Code and models will be released in two weeks.

##  Cite Uniformer

If you find this repository useful, please use the following BibTeX entry for citation.

```
wait for arxiv
```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Contributors and Contact Information

UniFormer is maintained by Kunchang Li.

For help or issues using UniFormer, please submit a GitHub issue.

For other communications related to UniFormer, please contact Kunchang Li (`kc.li@siat.ac.cn`). 
