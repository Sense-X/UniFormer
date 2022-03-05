# Image classification with UniFormer

We currenent release the code and models for:

- [x] ImageNet-1K pretraining

- [x] ImageNet-1K pretraining + [Token Labeling](https://arxiv.org/abs/2104.10858)

- [x] Large resolution fine-tuning



## Update

***03/6/2022***

Some models with `head_dim=64` are released, which can save memory cost for downstream tasks.

***01/19/2022***

1. Pretrained models on ImageNet-1K with [Token Labeling](https://arxiv.org/abs/2104.10858).
2. Large resolution fine-tuning.

***01/13/2022***

**\[Initial commits\]:** 

1. Pretrained models on ImageNet-1K.



## Model Zoo

### ImageNet-1K pretrained (224x224)

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/1-sqWoelz3agOG74VkNwadY8RHR28o4Wv?usp=sharing), [total_logs](https://drive.google.com/drive/folders/10MxmGoKxI3zMyCrzYaBDObCW7XgZn2tG?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (bdkq)](https://pan.baidu.com/s/1tLvcwGquAsQdGRAiKR0Jcg), [total_logs (ttub)](https://pan.baidu.com/s/1zFI-E-HRiYzmBtseB_tjEQ).

| Model                   | Top-1 | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                     |
| ----------------------- | ----- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| UniFormer-S             | 82.9  | 22M     | 3.6G  | [google](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [google](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) | [run.sh](exp/uniformer_small/run.sh)      |
| UniFormer-S†            | 83.4  | 24M     | 4.2G  | [google](https://drive.google.com/file/d/10IN5ULcjz0Ld_lDokkTGOSmRFXLzUkEs/view?usp=sharing) | [google](https://drive.google.com/file/d/10Onp1QzQ-Te90yEeQu1-xUcQyxPsQsU8/view?usp=sharing) | [run.sh](exp/uniformer_small_plus/run.sh) |
| UniFormer-B             | 83.8  | 50M     | 8.3G  | [google](https://drive.google.com/file/d/1-wT39QazTGELxgrQIu6J12D3qcla3hui/view?usp=sharing) | -                                                            | [run.sh](exp/uniformer_base/run.sh)       |
| UniFormer-B+Layer Scale | 83.9  | 50M     | 8.3G  | [google](https://drive.google.com/file/d/10FnwzMVgGL8bPO3EMZ7oG4vDvqnPoBK5/view?usp=sharing) | [google](https://drive.google.com/file/d/10RwaWu2EdC4Esnx5LSgGZT7iMCVOoc7b/view?usp=sharing) | [run.sh](exp/uniformer_base_ls/run.sh)    |

Though [Layer Scale](https://arxiv.org/abs/2103.17239) is helpful for training deep models, we meet some problems when fine-tuning on video datasets. Hence, we only use the models trained without it for video tasks.


Due to the model UniFormer-S† uses `head_dim=32`, which cause much memory cost for downstream tasks. We re-train these models with `head_dim=64`. All models are trained with 224x224 resolution.

| Model                   | Top-1 | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                     |
| ----------------------- | ----- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| UniFormer-S†            | 83.4  | 24M     | 4.2G  | [google](https://drive.google.com/file/d/178ipRvMBKeAP_Fzy6fEr2HQkIQWdgEY6/view?usp=sharing) | [google](https://drive.google.com/file/d/176GcmZz0t0A-1alBDWDX_BU_w2x0wACl/view?usp=sharing) | [run.sh](exp/uniformer_small_plus_dim64/run.sh) |


### ImageNet-1K pretrained with Token Labeling (224x224)

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/16iRLUaaxRHLzHg-LMJ3HUOK3dE8WHDPs?usp=sharing), [total_logs](https://drive.google.com/drive/folders/1672n8VVTFMdUKXk9nFr_avml7dk-cpq_?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (p05h)](https://pan.baidu.com/s/1y1TnUvaXteHZd-CMbm3vPA), [total_logs (wsvi)](https://pan.baidu.com/s/177eBUUoXTnHrSXwJ9wct3A).

We follow [LV-ViT](https://github.com/zihangJiang/TokenLabeling) to train our models with [Token Labeling](https://arxiv.org/abs/2104.10858). Please see [token_labeling](token_labeling)  for more details.

| Model                   | Top-1       | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                                        |
| ----------------------- | ----------- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S             | 83.4 (+0.5) | 22M     | 3.6G  | [google](https://drive.google.com/file/d/16wjfeyqQFZ2x9W91EXf0Py1bfv4Fydfr/view?usp=sharing) | [google](https://drive.google.com/file/d/16_l4JrV65uURG6e9NlqSiQcIhZGYCIit/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_small_tl_224/run.sh)   |
| UniFormer-S†            | 83.9 (+0.5) | 24M     | 4.2G  | [google](https://drive.google.com/file/d/16rgtEVEeX3jVTQaiLsojBQP5sRaj_2aZ/view?usp=sharing) | [google](https://drive.google.com/file/d/16SbjgFUfcqnQjcThRiw1LANk3oxzTgkI/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_small_plus_tl_224/run.sh) |
| UniFormer-B             | 85.1 (+1.3) | 50M     | 8.3G  | [google](https://drive.google.com/file/d/16nsTjawiYZuoCHBRarr58zA7vCFcmQmL/view?usp=sharing) | [google](https://drive.google.com/file/d/16eureelQrJEDnu53pNIWLkk2iqb5v0Sm/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_base_tl_224/run.sh)    |
| UniFormer-L+Layer Scale | 85.6        | 100M    | 12.6G | [google](https://drive.google.com/file/d/173e2qaJhOwbuC_DbNQSH47Oo_04mSfDj/view?usp=sharing) | [google](https://drive.google.com/file/d/16DyfgQUx8R5SquuqVB5Pll3nDIUJjRm1/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_large_ls_tl_224/run.sh) |


Due to the models UniFormer-S/S†/B use `head_dim=32`, which cause much memory cost for downstream tasks. We re-train these models with `head_dim=64`. All models are trained with 224x224 resolution.

| Model                   | Top-1       | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                                        |
| ----------------------- | ----------- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S             | 83.4 (+0.5) | 22M     | 3.6G  | [google](https://drive.google.com/file/d/17NC5qqAWCT2gI_jY_cJnFTOqRQTSwsoz/view?usp=sharing) | [google](https://drive.google.com/file/d/17K4egnbkwfuhQVA7zZDHf6fFYEJSeaJ_/view?usp=sharing) | [run.sh](exp/uniformer_small_dim64_tl_224/run.sh)   |
| UniFormer-S†            | 83.6 (+0.2) | 24M     | 4.2G  | [google](https://drive.google.com/file/d/17OfoDWt2_nA0BYN0OuLPpG8mHJDTluQQ/view?usp=sharing) | [google](https://drive.google.com/file/d/179gNIN8QOK1dCpWviB9GmLht80WO6qzT/view?usp=sharing) | [run.sh](exp/uniformer_small_plus_dim64_tl_224/run.sh) |
| UniFormer-B             | 84.8 (+1.0) | 50M     | 8.3G  | [google](https://drive.google.com/file/d/17MneG9CnZG6zBvXYUr4WUXoTYc4rjJj5/view?usp=sharing) | [google](https://drive.google.com/file/d/179gNIN8QOK1dCpWviB9GmLht80WO6qzT/view?usp=sharing) | [run.sh](exp/uniformer_base_dim64_tl_224/run.sh)    |

### Large resolution fine-tuning (384x384)

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/16iRLUaaxRHLzHg-LMJ3HUOK3dE8WHDPs?usp=sharing), [total_logs](https://drive.google.com/drive/folders/1672n8VVTFMdUKXk9nFr_avml7dk-cpq_?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (p05h)](https://pan.baidu.com/s/1y1TnUvaXteHZd-CMbm3vPA), [total_logs (wsvi)](https://pan.baidu.com/s/177eBUUoXTnHrSXwJ9wct3A).

We fine-tune the above models with Token Labeling on resolution of 384x384. Please see [token_labeling](token_labeling)  for more details.

| Model                   | Top-1 | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                                        |
| ----------------------- | ----- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S             | 84.6  | 22M     | 11.9G | [google](https://drive.google.com/file/d/16p4CXzuXC5J4_SJK67dsvnc8gTKmtKN1/view?usp=sharing) | [google](https://drive.google.com/file/d/16M8jhxW_frGnJQS_VMAF_rp9fcYssaYN/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_small_tl_384/run.sh)   |
| UniFormer-S†            | 84.9  | 24M     | 13.7G | [google](https://drive.google.com/file/d/16wJT87vTc43Dt1q2sXrxZRUuEdrNdFdk/view?usp=sharing) | [google](https://drive.google.com/file/d/16YVMpbHz1IdVi-HaocHpfHi_VAyFCvUk/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_small_plus_tl_384/run.sh) |
| UniFormer-B             | 86.0  | 50M     | 27.2G | [google](https://drive.google.com/file/d/16kZlIarIwf9ldkCdHVKTQfFmDgfkucQ7/view?usp=sharing) | [google](https://drive.google.com/file/d/16cPsL-Fo8XegiBQCy-aCozUHRAYWxFDd/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_base_tl_384/run.sh)    |
| UniFormer-L+Layer Scale | 86.3  | 100M    | 39.2G | [google](https://drive.google.com/file/d/174rcA6rNzYVG9Ya9ik-NwTGoxW1M79ez/view?usp=sharing) | [google](https://drive.google.com/file/d/16G-Bifg67v_8N_mPzi8-Ioo1qqAFPNN4/view?usp=sharing) | [run.sh](token_labeling/exp/uniformer_large_ls_tl_384/run.sh) |

## Usage

Our repository is built base on the [DeiT](https://github.com/facebookresearch/deit) repository, but we add some useful features:

1. Calculating accurate FLOPs and parameters with [fvcore](https://github.com/facebookresearch/fvcore) (see [check_model.py](check_model.py)).
2. Auto-resuming.
3. Saving best models and backup models.
4. Generating training curve (see [generate_tensorboard.py](generate_tensorboard.py)).

### Installation

- Clone this repo:

  ```shell
  git clone https://github.com/Sense-X/UniFormer.git
  cd UniFormer
  ```

- Install PyTorch 1.7.0+ and torchvision 0.8.1+

  ```shell
  conda install -c pytorch pytorch torchvision
  ```

- Install other packages

  ```shell
  pip install timm
  pip install fvcore
  ```

### Training

Simply run the training scripts in [exp](exp) as followed:

```shell
bash ./exp/uniformer_small/run.sh
```

If the training was interrupted abnormally, you can simply rerun the script for auto-resuming. Sometimes the checkpoint may not be saved properly, you should set the resumed model via `--reusme ${work_path}/ckpt/backup.pth`.

### Evaluation

Simply run the evaluating scripts in [exp](exp) as followed:

```shell
bash ./exp/uniformer_small/test.sh
```

It will evaluate the last model by default. You can set other models via `--resume`.

### Generate curves

You can generate the training curves as followed:

```shell
python3 generate_tensoboard.py
```

Note that you should install `tensorboardX`.

### Calculating FLOPs and Parameters

You can calculate the FLOPs and parameters via:

```shell
python3 check_model.py
```

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and the [DeiT](https://github.com/facebookresearch/deit) repository.

