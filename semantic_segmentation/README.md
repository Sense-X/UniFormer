# ADE20K Semantic segmentation with UniFormer

We currenent release the code and models for:

- [x] Semantic FPN

- [x] UperNet

  

## Updates


**05/22/2023**

Lightweight models with Semantic FPN are released.

**01/18/2022**

1. Models with Semantic FPN are released..

2. Models with UperNet are released.

   

## Model Zoo

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/13ceuU8sYa7xYWjdRbDsxIgpHcyoMqH-E?usp=sharing), [total_logs](https://drive.google.com/drive/folders/13hFb8mpJ-Sttfhha3uLXC65K3nmf5UVI?usp=sharing). 
>
> We also release the models on **Baidu Cloud**: [total_models (v5v2)](https://pan.baidu.com/s/12m09egivCGFkQosBRZb3NQ), [total_logs (mhc6)](https://pan.baidu.com/s/198xB3rXC19V-wlBloZyKZQ).

### Note

- All the models are pretrained on ImageNet-1K without [Token Labeling](https://arxiv.org/abs/2104.10858) and [Layer Scale](https://arxiv.org/abs/2103.17239). Reason can be found in [issue \#12](https://github.com/Sense-X/UniFormer/issues/12#issuecomment-1044001497).

### Semantic FPN

We follow [PVT](https://github.com/whai362/PVT/tree/v2/segmentation) to train our UniFormer with semantic FPN, please refer to [fpn_seg](fpn_seg).

|         Backbone          | Lr Schd | mIoU | #params | FLOPs |                            Model                             |                             Log                              |                            Shell                             |
| :----------------------- | :----- | :-- | :----- | :--- | :---------------------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------- |
| UniFormer-XXS |   80K   | 42.3 |   13.5M   | -  | [google](https://drive.google.com/file/d/1DPakdzr6G7n8t3_ZAFCx8wFX1r5h_V7-/view?usp=sharing) | [google](https://drive.google.com/file/d/1DJ9cseVGqk6aAe6dyI0jnL5F8X3GZ4bz/view?usp=sharing) | [run.sh](./fpn_seg/exp_light/fpn_xxs/run.sh)/[config](./fpn_seg/exp_light/fpn_xxs/config.py) |
| UniFormer-XS|   80K   | 44.4 |   19.7M   | -  |  [google](https://drive.google.com/file/d/1Dj3GiQLK7YZPKSgz_Uk8JL1J66XEvw0Y/view?usp=sharing) | [google](https://drive.google.com/file/d/1DLrCjhV2QnzuXjRz2NL5A57fNACCKEId/view?usp=sharing) | [run.sh](./fpn_seg/exp_light/fpn_xs/run.sh)/[config](./fpn_seg/exp_light/fpn_xs/config.py) |
| UniFormer-S<sub>h14</sub> |   80K   | 46.3 |   25M   | 172G  |                              -                               | [google](https://drive.google.com/file/d/166dIFxHt_xrkHMBe6vqy79TjQbfEQBW1/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_hybrid_small/run.sh)/[config](fpn_seg/exp/fpn_hybrid_small/config.py) |
| UniFormer-B<sub>h14</sub> |   80K   | 47.0 |   54M   | 328G  | [google](https://drive.google.com/file/d/14XokJDp1oYDV9jZnniMXyk3AnVfk1S1-/view?usp=sharing) | [google](https://drive.google.com/file/d/13wITOvL-iqIdyjhK8laEzsKvFDba-t9L/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_hybrid_base/run.sh)/[config](fpn_seg/exp/fpn_hybrid_base/config.py) |

For the models with the window size of 32, we train one model but test with different `test_config`.

|         Backbone          | Lr Schd | mIoU | #params | FLOPs |                            Model                             |                             Log                              |                            Shell                             |
| :----------------------- | :----- | :-- | :----- | :--- | :---------------------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------- |
| UniFormer-S<sub>w32</sub> |   80K   | 45.6 |   25M   | 183G  | [google](https://drive.google.com/file/d/14bEsZqMw35tmDgrvYA_dRMsSC0Z4TX2Y/view?usp=sharing) | [google](https://drive.google.com/file/d/14GpeCl8lfbsIcecyil14-_goGRNHd20w/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_small/run.sh)/[config](fpn_seg/exp/fpn_global_small/test_config_w32.py) |
| UniFormer-S<sub>h32</sub> |   80K   | 46.2 |   25M   | 199G  | [google](https://drive.google.com/file/d/14bEsZqMw35tmDgrvYA_dRMsSC0Z4TX2Y/view?usp=sharing) | [google](https://drive.google.com/file/d/14GpeCl8lfbsIcecyil14-_goGRNHd20w/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_small/run.sh)/[config](fpn_seg/exp/fpn_global_small/test_config_h32.py) |
|  UniFormer-S  |   80K   | 46.6 |   25M   | 247G  | [google](https://drive.google.com/file/d/14bEsZqMw35tmDgrvYA_dRMsSC0Z4TX2Y/view?usp=sharing) | [google](https://drive.google.com/file/d/14GpeCl8lfbsIcecyil14-_goGRNHd20w/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_small/run.sh)/[config](fpn_seg/exp/fpn_global_small/test_config_g.py) |
| UniFormer-B<sub>w32</sub> |   80K   | 47.0 |   54M   | 310G  | [google](https://drive.google.com/file/d/14dLoIcAUWjyo1d5XXuo1yb2jcslt_t4P/view?usp=sharing) | [google](https://drive.google.com/file/d/14CxzjtOkvxpCCXF3SeUd_MvJgNvZbYDt/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_base/run.sh)/[config](fpn_seg/exp/fpn_global_base/test_config_w32.py) |
| UniFormer-B<sub>h32</sub> |   80K   | 47.7 |   54M   | 350G  | [google](https://drive.google.com/file/d/14dLoIcAUWjyo1d5XXuo1yb2jcslt_t4P/view?usp=sharing) | [google](https://drive.google.com/file/d/14CxzjtOkvxpCCXF3SeUd_MvJgNvZbYDt/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_base/run.sh)/[config](fpn_seg/exp/fpn_global_base/test_config_h32.py) |
|  UniFormer-B  |   80K   | 48.0 |   54M   | 471G  | [google](https://drive.google.com/file/d/14dLoIcAUWjyo1d5XXuo1yb2jcslt_t4P/view?usp=sharing) | [google](https://drive.google.com/file/d/14CxzjtOkvxpCCXF3SeUd_MvJgNvZbYDt/view?usp=sharing) | [run.sh](fpn_seg/exp/fpn_global_base/run.sh)/[config](fpn_seg/exp/fpn_global_base/test_config_g.py) |

### UperNet

We follow [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) to train our UniFormer with UperNet.

|    Backbone     | Lr Schd | mIoU | MS mIoU | #params | FLOPs |                            Model                             |                             Log                              |                            Shell                             |
| :------------- | :----- | :-- | :-- | :----- | :--- | :---------------------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------- |
| UniFormer-S<sub>h14</sub> |  160K   | 46.9 | 48.0    |   52M   | 947G | - | [google](https://drive.google.com/file/d/13zp-FIVZkkzYrN7jI2ijXY0PmKtQAzfZ/view?usp=sharing) | [run.sh](exp/upernet_hybrid_small/run.sh)/[config](exp/upernet_hybrid_small/config.py) |
| UniFormer-B<sub>h14</sub> |  160K   | 48.9 | 50.0    |   80M   | 1085G | - | [google](https://drive.google.com/file/d/13r2ZLpyjLOglCY6YNtRe_GyWQVbjgtWp/view?usp=sharing) | [run.sh](exp/upernet_hybrid_base/run.sh)/[config](exp/upernet_hybrid_base/config.py) |

For the models with the window size of 32, we train one model but test with different `test_config`.

|    Backbone     | Lr Schd | mIoU | MS mIoU | #params | FLOPs |                            Model                             |                             Log                              |                            Shell                             |
| :------------- | :----- | :-- | :------- | :----- | :--- | :---------------------------------------------------------- | :---------------------------------------------------------- | :---------------------------------------------------------- |
| UniFormer-S<sub>w32</sub> |   160K   | 46.6 | 48.4 |   52M   | 939G | [google](https://drive.google.com/file/d/13hOneJiFqwEneUz1Zpqo1W2uv7BsoXGF/view?usp=sharing) | [google](https://drive.google.com/file/d/13xCDth4TuUfLsDa2MlM0UA74nR9FCHpg/view?usp=sharing) | [run.sh](exp/upernet_global_small/run.sh)/[config](exp/upernet_global_small/test_config_w32.py) |
| UniFormer-S<sub>h32</sub> |   160K   | 47.0 | 48.5 |   52M   | 955G | [google](https://drive.google.com/file/d/13hOneJiFqwEneUz1Zpqo1W2uv7BsoXGF/view?usp=sharing) | [google](https://drive.google.com/file/d/13xCDth4TuUfLsDa2MlM0UA74nR9FCHpg/view?usp=sharing) | [run.sh](exp/upernet_global_small/run.sh)/[config](exp/upernet_global_small/test_config_h32.py) |
|  UniFormer-S  |   160K   | 47.6 | 48.5 |   52M   | 1004G | [google](https://drive.google.com/file/d/13hOneJiFqwEneUz1Zpqo1W2uv7BsoXGF/view?usp=sharing) | [google](https://drive.google.com/file/d/13xCDth4TuUfLsDa2MlM0UA74nR9FCHpg/view?usp=sharing) | [run.sh](exp/upernet_global_small/run.sh)/[config](exp/upernet_global_small/test_config_g.py) |
| UniFormer-B<sub>w32</sub> |   160K   | 49.1 | 50.6 |   80M   | 1066G | [google](https://drive.google.com/file/d/14bEgmFbTijBoTKTwny_aCJlARs6z31mP/view?usp=sharing) | [google](https://drive.google.com/file/d/13yqQKCksnQ07685f11qb0bNzKwfeF7Cq/view?usp=sharing) | [run.sh](exp/upernet_global_base/run.sh)/[config](exp/upernet_global_base/test_config_w32.py) |
| UniFormer-B<sub>h32</sub> |   160K   | 49.5 | 50.7 |   80M   | 1106G | [google](https://drive.google.com/file/d/14bEgmFbTijBoTKTwny_aCJlARs6z31mP/view?usp=sharing) | [google](https://drive.google.com/file/d/13yqQKCksnQ07685f11qb0bNzKwfeF7Cq/view?usp=sharing) | [run.sh](exp/upernet_global_base/run.sh)/[config](exp/upernet_global_base/test_config_h32.py) |
|  UniFormer-B  |   160K   | 50.0 | 50.8 |   80M   | 1227G | [google](https://drive.google.com/file/d/14bEgmFbTijBoTKTwny_aCJlARs6z31mP/view?usp=sharing) | [google](https://drive.google.com/file/d/13yqQKCksnQ07685f11qb0bNzKwfeF7Cq/view?usp=sharing) | [run.sh](exp/upernet_global_base/run.sh)/[config](exp/upernet_global_base/test_config_g.py) |

## Usage

### Installation

Please refer to [get_started](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Training

1. Download the pretrained models in our repository.

2. Simply run the training scripts in [exp](exp) as followed:

   ```shell
   bash ./exp/upernet_global_small/run.sh
   ```

   Or you can train other models as follower:

   ```shell
   # single-gpu training
   python tools/train.py <CONFIG_FILE> --options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments]
   
   # multi-gpu training
   tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --options model.backbone.pretrained_path=<PRETRAIN_MODEL> [other optional arguments] 
   ```

**[Note]:**

- We use hybrid MHRA during testing and set the corresponding hyperparameters in the `config.py`:

  ```python
  window: False, # whether use window MHRA
  hybrid: True, # whether use hybrid MHRA
  window_size: 32, # size of window (>=14)
  ```

- To avoid **out of memory**, we use `torch.utils.checkpoint`  in the `config.py`:

  ```python
  use_checkpoint=True, # whether use checkpoint
  checkpoint_num=[0, 0, 2, 0], # index for using checkpoint in every stage
  ```

### Testing

```shell
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval mIoU
```

## Acknowledgement

This repository is built based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) and [PVT](https://github.com/whai362/PVT/tree/v2/segmentation) repository.