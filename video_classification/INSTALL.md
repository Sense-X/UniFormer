# Installation

## Requirements
- Python >= 3.8
- Numpy
- PyTorch >= 1.9 (Acceleration for 3D depth-wise convolution)
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`
- Decord: `pip install decord`

Different from the  [SlowFast](https://github.com/facebookresearch/SlowFast) repository, we remove some codes using  [Detectron2](https://github.com/facebookresearch/detectron2) for easy installation, which are about detection and visulazation. If you want to used them, please follow the [SlowFast](https://github.com/facebookresearch/SlowFast) repository.

## Build UniFormer
After having the above dependencies, run:
```shell
git clone https://github.com/Sense-X/UniFormer
cd UniFormer/video_classification
python setup.py build develop
```

