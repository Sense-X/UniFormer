# Video classification with UniFormer

We currenent release the code and models for:

- [x] Kintics-400
- [x] Kinetics-600
- [x] Something-Something V1
- [x] Something-Something V2




## Update


***05/21/2022***

Lightweight models are released, which surpass [X3D](https://arxiv.org/abs/2004.04730) and [MoViNet](https://arxiv.org/abs/2103.11511).

***01/13/2022***

Pretrained models on Kinetics-400, Kinetics-600, Something-Something V1&V2 d



## Model Zoo

> The followed models and logs can be downloaded on **Google Drive**: [total_models](https://drive.google.com/drive/folders/18LRT7FoCtH6LNGl6QNag58XEYoV_htbl?usp=sharing), [total_logs](https://drive.google.com/drive/folders/10W0RZiqy7fS4id_5uAkcIQdbGEtdSfvp?usp=sharing).
>
> We also release the models on **Baidu Cloud**: [total_models (gphp)](https://pan.baidu.com/s/1lewLGZvPcf9ff3BCT4jpGw), [total_logs (q5bw)](https://pan.baidu.com/s/1jii4wpgxrnrY_YcZXcUHHw).

### Note

- All the `config.yaml` in our `exp` are **NOT** the training config actually used, since some hyperparameters are **changed** in the `run.sh` or `test.sh`.
- All the models are pretrained on ImageNet-1K without [Token Labeling](https://arxiv.org/abs/2104.10858) and [Layer Scale](https://arxiv.org/abs/2103.17239). You can find those pre-trained models in [image_classification](../image_classification). Reason can be found in [issue \#12](https://github.com/Sense-X/UniFormer/issues/12#issuecomment-1044001497).
- \#Frame = \#input_frame x \#crop x \#clip
- \#input_frame means how many frames are input for model per inference
- \#crop means spatial crops (e.g., 3 for left/right/center)
- \#clip means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

### Kinetics-400

| Model         | #Frame | Resolution | FLOPs | Top1 | Model      | Log        | Shell | 
| ------------- | ------ | ---------- | ----- | -----| ---------- | ---------- | ----- |
| UniFormer-XXS | 4x1x1  | 128        | 1.0G  | 63.2 | [google](https://drive.google.com/file/d/1EFBxDS2m4mpk_sYejnPuGSDRH_1ELDhx/view?usp=sharing) | [google](https://drive.google.com/file/d/1EZK8h0lUswSISzRqW4ymS0nCzOfmiT0O/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs4_128_k400/run.sh)/[config](./exp_light/uniformer_xxs4_128_k400/config.yaml) |
| UniFormer-XXS | 4x1x1  | 160        | 1.6G  | 65.8 | [google](https://drive.google.com/file/d/1DtTFT7FZ_kUjxqXAYyWhfXSr_EIHQi8a/view?usp=sharing) | [google](https://drive.google.com/file/d/1EUemnw6KRa8ydf7dJS-LqESM3F51alvq/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs4_160_k400/run.sh)/[config](./exp_light/uniformer_xxs4_160_k400/config.yaml) |
| UniFormer-XXS | 8x1x1  | 128        | 2.0G  | 68.3 | [google](https://drive.google.com/file/d/1E9VwV-XJQCmo9nBhxLxa9DdphZiIg38z/view?usp=sharing) | [google](https://drive.google.com/file/d/1Eoh3tMWeef5IkM5TpOEsfFvTTzAxbNws/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs8_128_k400/run.sh)/[config](./exp_light/uniformer_xxs8_128_k400/config.yaml) |
| UniFormer-XXS | 8x1x1  | 160        | 3.3G  | 71.4 | [google](https://drive.google.com/file/d/1Dxq82X8mBsfVOzij0HSeb1g4sdez71FM/view?usp=sharing) | [google](https://drive.google.com/file/d/1ESTw2R3ldecynk6QdtMDDea0OeH6RvF7/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs8_160_k400/run.sh)/[config](./exp_light/uniformer_xxs8_160_k400/config.yaml) |
| UniFormer-XXS | 16x1x1 | 128        | 4.2G  | 73.3 | [google](https://drive.google.com/file/d/1DngSOmw0rStnYZlaCnz1xII6CU2d2R_j/view?usp=sharing) | [google](https://drive.google.com/file/d/1ETjHEKGZNvAFCnDijA1gr34NHbliGvhR/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs16_128_k400/run.sh)/[config](./exp_light/uniformer_xxs16_128_k400/config.yaml) |
| UniFormer-XXS | 16x1x1 | 160        | 6.9G  | 75.1 | [google](https://drive.google.com/file/d/1DxqrNHb4h1ZKnEFaqbkKJAXQsyIFeD7b/view?usp=sharing) | [google](https://drive.google.com/file/d/1EVHCoy9tHYfn7f-9bG_O1tFtcLhYKVVC/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs16_160_k400/run.sh)/[config](./exp_light/uniformer_xxs16_160_k400/config.yaml) |
| UniFormer-XXS | 32x1x1 | 160        | 15.4G | 77.9 | [google](https://drive.google.com/file/d/1EIVXRYn5UIkLaJGDpTBxTcyog_dKbcZe/view?usp=sharing) | [google](https://drive.google.com/file/d/1ENZYQ_T7_X4cPiF3HBS6haTWBcgMp7TP/view?usp=sharing) | [run.sh](./exp_light/uniformer_xxs32_160_k400/run.sh)/[config](./exp_light/uniformer_xxs32_160_k400/config.yaml) |
| UniFormer-XS  | 32x1x1 | 192        | 34.2G | 78.6 | [google](https://drive.google.com/file/d/1DtdX-keSB-RkH23KqtQH36IMOUmol1kw/view?usp=sharing) | [google](https://drive.google.com/file/d/1EWFk-INNe5pnHVz3przqADvqPJS6nlCk/view?usp=sharing) | [run.sh](./exp_light/uniformer_xs32_192_k400/run.sh)/[config](./exp_light/uniformer_xs32_192_k400/config.yaml) |

We adopt sparse sampling method for lightweight models. And to avoid loss NAN , we use the following techiniques:
- Close mixed precision training.
- Use weaker data augmentation.
- Add [Layer Scale](https://arxiv.org/abs/2103.17239).


| Model       | #Frame | Sampling Stride | FLOPs | Top1 | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | ------ | --------------- | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | 8x1x4  | 8            | 70G   | 78.4 | [google](https://drive.google.com/file/d/1-kitVTkWErHXI_x-sLItvK9-sOmb53j6/view?usp=sharing) | [google](https://drive.google.com/file/d/12k73hKC4TKHf895UJz2F7KeCIYOR5kHJ/view?usp=sharing) | [run.sh](exp/uniformer_s8x8_k400/run.sh)/[config](exp/uniformer_s8x8_k400/config.yaml) |
| UniFormer-S | 16x1x4 | 4            | 167G  | 80.8 | [google](https://drive.google.com/file/d/1-jGZIIuTM5IYIpqkNTojhGhfMc7Gu9Kt/view?usp=sharing) | [google](https://drive.google.com/file/d/12WFVym5nKX4cObGsZQrjSxKotK0ntIkl/view?usp=sharing) | [run.sh](exp/uniformer_s16x4_k400/run.sh)/[config](exp/uniformer_s16x4_k400/config.yaml) |
| UniFormer-S | 16x1x4 | 8            | 167G  | 80.8 | [google](https://drive.google.com/file/d/174muWZJcZHieABDZMKiJ0tNeFLoEkJLx/view?usp=sharing) | [google](https://drive.google.com/file/d/12t9i4G9F3Po0k7380n3oFziruBTR5Ksx/view?usp=sharing) | [run.sh](exp/uniformer_s16x8_k400/run.sh)/[config](exp/uniformer_s16x8_k400/config.yaml) |
| UniFormer-S | 32x1x4 | 4            | 438G  | 82.0 | -                                                            | [google](https://drive.google.com/file/d/12_EVFS3lEJp4_VdzYj87-7tlLgHX-pZE/view?usp=sharing) | [run.sh](exp/uniformer_s32x4_k400/run.sh)/[config](exp/uniformer_s32x4_k400/config.yaml) |
| UniFormer-B | 8x1x4  | 8            | 161G  | 79.8 | [google](https://drive.google.com/file/d/1hBxGU-QE8hGRBjSNCymJ7Au06LhluvTv/view?usp=sharing) | [google](https://drive.google.com/file/d/12cJfcsqsTb7D0lYfP-M1D3hQurd-dOyC/view?usp=sharing) | [run.sh](exp/uniformer_b8x8_k400/run.sh)/[config](exp/uniformer_b8x8_k400/config.yaml) |
| UniFormer-B | 16x1x4 | 4            | 387G  | 82.0 | [google](https://drive.google.com/file/d/15ipzFaZdTt-aHv7jcr_h_vgYD2X7D6AD/view?usp=sharing) | [google](https://drive.google.com/file/d/11y_d_GZvKsbYiBxkHdf42vAqEfWZ75Ei/view?usp=sharing) | [run.sh](exp/uniformer_b16x4_k400/run.sh)/[config](exp/uniformer_b16x4_k400/config.yaml) |
| UniFormer-B | 16x1x4 | 8            | 387G  | 81.7 | [google](https://drive.google.com/file/d/1VoZcpFZbk9lEyRQ7gqRgT4CsnVt9g7uk/view?usp=sharing) | [google](https://drive.google.com/file/d/11QJbi00DT01InNq5nVsGr0RoC9XD5roS/view?usp=sharing) | [run.sh](exp/uniformer_b16x8_k400/run.sh)/[config](exp/uniformer_b16x8_k400/config.yaml) |
| UniFormer-B | 32x1x4 | 4            | 1036G | 82.9 | [google](https://drive.google.com/file/d/1gHxx8cr1H0CNvngj1kB9ArgcPfJtqjCV/view?usp=sharing) | [google](https://drive.google.com/file/d/124QnS3N3SskdrscqDi5V_NO-OpU6cM1N/view?usp=sharing) | [run.sh](exp/uniformer_b32x4_k400/run.sh)/[config](exp/uniformer_b32x4_k400/config.yaml) |

### Kinetics-600

| Model       | #Frame | Sampling Stride | FLOPs | Top1  | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | ------ | --------------- | ----- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | 16x1x4 | 4            | 167G  | 82.8  | [google](https://drive.google.com/file/d/1-dqzjm5RZVspWHQLRD4S1vo4_6jyVltb/view?usp=sharing) | [google](https://drive.google.com/file/d/11zBT4kaUlpvm0uCF3CH4jjC7-ZL4NBNJ/view?usp=sharing) | [run.sh](exp/uniformer_s16x4_k600/run.sh)/[config](exp/uniformer_s16x4_k600/config.yaml) |
| UniFormer-S | 16x1x4 | 8            | 167G  | 82.7  | [google](https://drive.google.com/file/d/1-c835NJjg_015WxLBQF-1cPdQsyWMWfY/view?usp=sharing) | [google](https://drive.google.com/file/d/11bz_PdQfdiyN2WwwOyi8dUStE89Z3a2V/view?usp=sharing) | [run.sh](exp/uniformer_s16x8_k600/run.sh)/[config](exp/uniformer_s16x8_k600/config.yaml) |
| UniFormer-B | 16x1x4 | 4            | 387G  | 84.0  | [google](https://drive.google.com/file/d/1nBQCBCsCflhZ4OGUoGB3pAkc_GDau7vd/view?usp=sharing) | [google](https://drive.google.com/file/d/12WzNO_1MgPRY3ukLpR4mNMgerp7r7NLA/view?usp=sharing) | [run.sh](exp/uniformer_b16x4_k600/run.sh)/[config](exp/uniformer_b16x4_k600/config.yaml) |
| UniFormer-B | 16x1x4 | 8            | 387G  | 83.4  | [google](https://drive.google.com/file/d/1aidHdSd-h65-_EUkfNOdMMw0ZB8G5maY/view?usp=sharing) | [google](https://drive.google.com/file/d/10wd98Tabutnc0W7c-5p63mwdF_bV-VNl/view?usp=sharing) | [run.sh](exp/uniformer_b16x8_k600/run.sh)/[config](exp/uniformer_s16x8_k600/config.yaml) |
| UniFormer-B | 32x1x4 | 4            | 1036G | 84.5* | [google](https://drive.google.com/file/d/1-DwdVf8w8lYj-iFpU40pfEpog9VE5PQB/view?usp=sharing) | [google](https://drive.google.com/file/d/12YDYVKANqk8OJnw8UbOnGema3mf7kU4P/view?usp=sharing) | [run.sh](exp/uniformer_b32x4_k600/run.sh)/[config](exp/uniformer_b32x4_k600/config.yaml) |

\* Since Kinetics-600 is too large to train (>1 month in single node with 8 A100 GPUs), we provide model trained in multi node (around 2 weeks with 32 V100 GPUs), but the result is lower due to the lack of tuning hyperparameters.

For Multi-node training, please install [submitit](https://github.com/facebookincubator/submitit) or follow the training scripts in our [UniFormerV2](https://github.com/OpenGVLab/UniFormerV2).

### Something-Something V1

| Model       | Pretrain | #Frame | FLOPs | Top1 | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | -------- | ------ | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | K400     | 16x3x1 | 125G  | 57.2 | [google](https://drive.google.com/file/d/1-bdFsmXt2Ztf43lrVgmqS8_VYJ8tub6J/view?usp=sharing) | [google](https://drive.google.com/file/d/12iND3FmTF5kzAqZhRv9Xti0pLm7sXYlZ/view?usp=sharing) | [run.sh](exp/uniformer_s16_sthv1_prek400/run.sh)/[config](exp/uniformer_s16_sthv1_prek400/config.yaml) |
| UniFormer-S | K600     | 16x3x1 | 125G  | 57.6 | [google](https://drive.google.com/file/d/1-YbYGjM15MDtQnrmCeZ9H2ybNdYTFkgO/view?usp=sharing) | [google](https://drive.google.com/file/d/12fJIWHY5svqYAOmYPrgEqDYkpaac-LKT/view?usp=sharing) | [run.sh](exp/uniformer_s16_sthv1_prek600/run.sh)/[config](exp/uniformer_s16_sthv1_prek600/config.yaml) |
| UniFormer-S | K400     | 32x3x1 | 329G  | 58.8 | [google](https://drive.google.com/file/d/1-Y5ADssqM2C8o_1jix48fQM788mVGWQ3/view?usp=sharing) | [google](https://drive.google.com/file/d/122TW8849pBNfwH9PlGKhvJlYjigHeY3C/view?usp=sharing) | [run.sh](exp/uniformer_s32_sthv1_prek400/run.sh)/[config](exp/uniformer_s32_sthv1_prek400/config.yaml) |
| UniFormer-S | K600     | 32x3x1 | 329G  | 59.9 | [google](https://drive.google.com/file/d/1-XEgaGqXGp1EGzFl63ctApnvU54X27WP/view?usp=sharing) | [google](https://drive.google.com/file/d/11fG8QyYMW2B03YycwjRnPS8-2ZrQH6ae/view?usp=sharing) | [run.sh](exp/uniformer_s32_sthv1_prek600/run.sh)/[config](exp/uniformer_s32_sthv1_prek600/config.yaml) |
| UniFormer-B | K400     | 16x3x1 | 290G  | 59.1 | [google](https://drive.google.com/file/d/1uMq4W5lf17vOHUmccNjYLeTMqaWHSh6v/view?usp=sharing) | [google](https://drive.google.com/file/d/11mKvtyeXknDIkEaeYi_ysUVKEreJtUmP/view?usp=sharing) | [run.sh](exp/uniformer_b16_sthv1_prek400/run.sh)/[config](exp/uniformer_b16_sthv1_prek400/config.yaml) |
| UniFormer-B | K600     | 16x3x1 | 290G  | 58.8 | [google](https://drive.google.com/file/d/1xpiL8zEDWYHFR55gOsbQiI41YEjTD89J/view?usp=sharing) | [google](https://drive.google.com/file/d/11dz7vWE6qlHUaAHbVfHcskPNpV-JE_8i/view?usp=sharing) | [run.sh](exp/uniformer_b16_sthv1_prek600/run.sh)/[config](exp/uniformer_b16_sthv1_prek600/config.yaml) |
| UniFormer-B | K400     | 32x3x1 | 777G  | 60.9 | [google](https://drive.google.com/file/d/18A3FuUUbdQ8ehMQ0D78qTTtm6YmIVWaO/view?usp=sharing) | [google](https://drive.google.com/file/d/11BBsG0DvHp1RipclvOBe5JcSC5qzNOss/view?usp=sharing) | [run.sh](exp/uniformer_b32_sthv1_prek400/run.sh)/[config](exp/uniformer_b32_sthv1_prek400/config.yaml) |
| UniFormer-B | K600     | 32x3x1 | 777G  | 61.0 | [google](https://drive.google.com/file/d/1xE1aJYAYFOlXOkpiIbJFHu_jSrr8reKm/view?usp=sharing) | [google](https://drive.google.com/file/d/11G6XBr49guaFPwKpyqa_CEnclPdI2N5w/view?usp=sharing) | [run.sh](exp/uniformer_b32_sthv1_prek600/run.sh)/[config](exp/uniformer_b32_sthv1_prek600/config.yaml) |

### Something-Something V2

| Model       | Pretrain | #Frame | FLOPs | Top1 | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | -------- | ------ | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | K400     | 16x3x1 | 125G  | 67.7 | [google](https://drive.google.com/file/d/1-Wq22024HB7tT8kPX6eSMn2GUNMI-_gY/view?usp=sharing) | [google](https://drive.google.com/file/d/12Hc63B0e_AqeB4Av3dW0hCQq-CstpNy1/view?usp=sharing) | [run.sh](exp/uniformer_s16_sthv2_prek400/run.sh)/[config](exp/uniformer_s16_sthv2_prek400/config.yaml) |
| UniFormer-S | K600     | 16x3x1 | 125G  | 69.4 | [google](https://drive.google.com/file/d/1-V1mQp4fIIYhaXUWCgZXkJ0H6ne0Z8rz/view?usp=sharing) | [google](https://drive.google.com/file/d/12PRSCa4G8QFD6xPDd0mUBm05AVm23LIk/view?usp=sharing) | [run.sh](exp/uniformer_s16_sthv2_prek600/run.sh)/[config](exp/uniformer_s16_sthv2_prek600/config.yaml) |
| UniFormer-S | K400     | 32x3x1 | 329G  | 69.0 | [google](https://drive.google.com/file/d/1-5Q5S-NRmHFkIBygaRYpJLT7aI9cpmky/view?usp=sharing) | [google](https://drive.google.com/file/d/12bOBMlCkXPYxyTCaHq6u_-_VqvmIyVtp/view?usp=sharing) | [run.sh](exp/uniformer_s32_sthv2_prek400/run.sh)/[config](exp/uniformer_s32_sthv2_prek400/config.yaml) |
| UniFormer-S | K600     | 32x3x1 | 329G  | 70.4 | [google](https://drive.google.com/file/d/1-1K5R3Ed4t8d_DZnJ62xysIt1dBNmx96/view?usp=sharing) | [google](https://drive.google.com/file/d/12VlpfLnlGJP7O_RozN9ZGrbi7eiXup9x/view?usp=sharing) | [run.sh](exp/uniformer_s32_sthv2_prek600/run.sh)/[config](exp/uniformer_s32_sthv2_prek600/config.yaml) |
| UniFormer-B | K400     | 16x3x1 | 290G  | 70.4 | [google](https://drive.google.com/file/d/1s78JBpnykyJ7pYEGRDK8BZL9seIRIUlC/view?usp=sharing) | [google](https://drive.google.com/file/d/11PU7v4YA7-SaOS-g8GLbmfZ-3s76qhmk/view?usp=sharing) | [run.sh](exp/uniformer_b16_sthv2_prek400/run.sh)/[config](exp/uniformer_b16_sthv2_prek400/config.yaml) |
| UniFormer-B | K600     | 16x3x1 | 290G  | 70.2 | [google](https://drive.google.com/file/d/16nCBGWzMM8n8Vr_EY0UmvgzLG0S_QYef/view?usp=sharing) | [google](https://drive.google.com/file/d/10aaqr20Vp_UAiC4GvuiK-Uo0eyneNl4b/view?usp=sharing) | [run.sh](exp/uniformer_b16_sthv2_prek600/run.sh)/[config](exp/uniformer_b16_sthv2_prek600/config.yaml) |
| UniFormer-B | K400     | 32x3x1 | 777G  | 71.1 | [google](https://drive.google.com/file/d/1-rpMARXnyvyj6YUJkIvVqtna86egpjoS/view?usp=sharing) | [google](https://drive.google.com/file/d/11SA-64b-3ty4BC8w_vXtXYx5Sjn5nYPx/view?usp=sharing) | [run.sh](exp/uniformer_b32_sthv2_prek400/run.sh)/[config](exp/uniformer_b32_sthv2_prek400/config.yaml) |
| UniFormer-B | K600     | 32x3x1 | 777G  | 71.2 | [google](https://drive.google.com/file/d/1-lkRB3546l_ECuObN_IrHws1C9a-qKPr/view?usp=sharing) | [google](https://drive.google.com/file/d/11Gzhwz5GAqcMxUA3JOUeeZnjCcv09Afy/view?usp=sharing) | [run.sh](exp/uniformer_b32_sthv2_prek600/run.sh)/[config](exp/uniformer_b32_sthv2_prek600/config.yaml) |

### UCF101

| Model       | #Frame | Sampling Stride | FLOPs | Top1 | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | ------ | --------------- | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | 16x3x5  | 4            | 625G   | 98.3 | [google](https://drive.google.com/file/d/1Ey8JIGcP8G2H0bKxyNnD61lvHV1eerVt/view?usp=sharing) | [google](https://drive.google.com/file/d/1F0pK6VaJOOxtTb1gwT8OxXUuMSN3LGf-/view?usp=sharing) | [run.sh](./exp/uniformer_s16x4_ucf101_prek400/run.sh)/[config](./exp/uniformer_s16x4_ucf101_prek400/config.yaml) |

### HMDB51

| Model       | #Frame | Sampling Stride | FLOPs | Top1 | Model                                                        | Log                                                          | Shell                                                        |
| ----------- | ------ | --------------- | ----- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| UniFormer-S | 16x3x5  | 4            | 625G   | 77.5 | [google](https://drive.google.com/file/d/1Es4xtcEvQSomnbp6Eo7TC3JejIwHf5yS/view?usp=sharing) | [google](https://drive.google.com/file/d/1EzTnQrHslHV4wpbGRLVeBvm_HjYeCma_/view?usp=sharing) | [run.sh](./exp/uniformer_s16x4_hmdb51_prek400/run.sh)/[config](./exp/uniformer_s16x4_hmdb51_prek400/config.yaml) |


## Usage

### Installation

Please follow the installation instructions in [INSTALL.md](INSTALL.md). You may follow the instructions in [DATASET.md](DATASET.md) to prepare the datasets.

### Training

1. Download the pretrained models in our repository.

2. Simply run the training scripts in [exp](exp) as followed:

   ```shell
   bash ./exp/uniformer_s8x8_k400/run.sh
   ```

**[Note]:**

- Due to some bugs in the [SlowFast](https://github.com/facebookresearch/SlowFast) repository, the program will be terminated in the final testing.

- During training, we follow the [SlowFast](https://github.com/facebookresearch/SlowFast) repository and randomly crop videos for validation. For accurate testing, please follow our testing scripts.

- For more config details, you can read the comments in `slowfast/config/defaults.py`.

- To avoid **out of memory**, you can use `torch.utils.checkpoint` (in `config.yaml` or `run.sh`):

  ```yaml
  MODEL.USE_CHECKPOINT True # whether use checkpoint
  MODEL.CHECKPOINT_NUM [0, 0, 4, 0] # index for using checkpoint in every stage
  ```

### Testing

We provide testing example as followed:

```shell
bash ./exp/uniformer_s8x8_k400/test.sh
```

Specifically, we need to create our new config for testing and run multi-crop/multi-clip test:

1. Copy the training config file `config.yaml` and create new testing config `test.yaml`.

2. Change the hyperparameters of data (in `test.yaml` or `test.sh`):

   ```yaml
   DATA:
     TRAIN_JITTER_SCALES: [224, 224]
     TEST_CROP_SIZE: 224
   ```

3. Set the number of crops and clips (in `test.yaml` or `test.sh`):

   **Multi-clip testing for Kinetics**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 4
   TEST.NUM_SPATIAL_CROPS 1
   ```

   **Multi-crop testing for Something-Something**

   ```shell
   TEST.NUM_ENSEMBLE_VIEWS 1
   TEST.NUM_SPATIAL_CROPS 3
   ```

4. You can also set the checkpoint path via:

   ```shell
   TEST.CHECKPOINT_FILE_PATH your_model_path
   ```

##  Cite Uniformer

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{li2022uniformer,
      title={Uniformer: Unified Transformer for Efficient Spatiotemporal Representation Learning}, 
      author={Kunchang Li and Yali Wang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
      year={2022},
      eprint={2201.04676},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

This repository is built based on [SlowFast](https://github.com/facebookresearch/SlowFast) repository.

