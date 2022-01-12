# Image classification with UniFormer

We currenent release the code and models for:

- [x] ImageNet-1K pretraining

- [ ] ImageNet-1K pretraining + Token Labeling

- [ ] Large resolution fine-tuning



## Update

***01/13/2022***

**\[Initial commits\]:** 

1. Pretrained models on ImageNet-1K



## Model Zoo

- ImageNet-1K pretrained

- Resolution 224x224

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/1-sqWoelz3agOG74VkNwadY8RHR28o4Wv?usp=sharing), [total_logs](https://drive.google.com/drive/folders/10MxmGoKxI3zMyCrzYaBDObCW7XgZn2tG?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (bdkq)](https://pan.baidu.com/s/1tLvcwGquAsQdGRAiKR0Jcg), [total_logs (ttub)](https://pan.baidu.com/s/1zFI-E-HRiYzmBtseB_tjEQ).

| Model                   | Top-1 | #Param. | FLOPs | Model                                                        | Log                                                          | Shell                                     |
| ----------------------- | ----- | ------- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| UniFormer-S             | 82.9  | 22M     | 3.6G  | [google](https://drive.google.com/file/d/1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD/view?usp=sharing) | [google](https://drive.google.com/file/d/10ThKb9YOpCiHW8HL10dRuZ0lQSPJidO7/view?usp=sharing) | [run.sh](exp/uniformer_small/run.sh)      |
| UniFormer-S†            | 83.4  | 24M     | 4.2G  | [google](https://drive.google.com/file/d/10IN5ULcjz0Ld_lDokkTGOSmRFXLzUkEs/view?usp=sharing) | [google](https://drive.google.com/file/d/10Onp1QzQ-Te90yEeQu1-xUcQyxPsQsU8/view?usp=sharing) | [run.sh](exp/uniformer_small_plus/run.sh) |
| UniFormer-B             | 83.8  | 50M     | 8.3G  | [google](https://drive.google.com/file/d/1-wT39QazTGELxgrQIu6J12D3qcla3hui/view?usp=sharing) | -                                                            | [run.sh](exp/uniformer_base/run.sh)       |
| UniFormer-B+Layer Scale | 83.9  | 50M     | 8.3G  | [google](https://drive.google.com/file/d/10FnwzMVgGL8bPO3EMZ7oG4vDvqnPoBK5/view?usp=sharing) | [google](https://drive.google.com/file/d/10RwaWu2EdC4Esnx5LSgGZT7iMCVOoc7b/view?usp=sharing) | [run.sh](exp/uniformer_base_ls/run.sh)    |

Though [Layer Scale](https://arxiv.org/abs/2103.17239) is helpful for training deep models, we meet some problems when fine-tuning on video datasets. Hence, we only use the models trained without it.

## Usage

Our repository is built base on the [DeiT](https://github.com/facebookresearch/deit) repository, but we add some useful features:

1. Calculating accurate FLOPs and parameters with [fvcore](https://github.com/facebookresearch/fvcore) (see [check_model.py](check_model.py)).
2. Auto-resuming.
3. Saving best models and backup models.
4. Generating training curve (see [generate_tensorboard.py](generate_tensorboard.py)).

## Installation

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

