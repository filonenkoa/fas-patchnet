# PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition

This repository implements PatchNet from paper [PatchNet: A Simple Face Anti-Spoofing Framework via Fine-Grained Patch Recognition](https://arxiv.org/abs/2203.14325) (UNOFFICIAL)

After carefully examining the [original repository](https://github.com/doantienthongbku/Implementation-patchnet), I have identified a number of missing features that I believe would greatly benefit its functionality. In light of this, I have made the decision to fork the repository and implement these features. Specifically, I incorporated distributed data parallel as well as training and validation for multiple datasets.


## Improvements to the original repository
- [x] More augmentations (torchvision ➞ albumentations)
- [x] TurboJPEG support for faster image decoding
- [x] DDP support
- [x] Multiple datasets training
- [x] Utility to convert datasets
- [x] Compute FAS-related metrics (ACER, etc.)
- [x] Incorporate loss into a model (the whole inference can be exported to a single ONNX file)
- [x] Telegram reports
- [x] Compute metrics for each val dataset separately
- [x] Split validation into miltiple GPUs
- [ ] Balanced sampler suitable for DDP
- [ ] Conversion to ONNX


## Installation
```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

When using EfficientFormerV2 model, put pretrained weights to `weights/efficientformerv2`

### Telegram reports
This repository supports sending messages after each epoch to a telegram bot.\
To make it work, create `.env` file with Telegram bot token and chat ids:
```bash
$ cp .env.template .env
$ nano .env
```

## Usage
### Data preparation
```
datasets
    |---images
    |     |--img1
    |     |--img2
    |     |...
    |---train.csv
    |---val.csv
    |---test.csv
```
with [*.csv] having format (label only has 2 classes: 0-Spoofing, 1-Bonafide):
```
image_name      |  label
img_name1.jpg   |    0
img_name2.png   |    1
...
```
`image_name` is the relative path to the image from the locations of the *.csv file.\
One can find utility to convert exisiting images dataset into format supported by current repository in `utils/dataset_preparation/prepare_dataset.py`

### Training
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 tool/train.py --config config.yaml
```
`nproc_per_node` is the number of GPUs you want to use.


### Testing
`This step has not been remade by me yet.`\
Go to tool/test.py and fix saved_name to your path to checkpoint \
Run
```bash
python3 test.py
```
## Original repository contributors
- Tien Thong Doan
- Minh Chau Nguyen
- Minh Hung Nguyen

## This repository contributors
- Alexander Filonenko
