# SlowFastNetworks for Pl
PyTorch Lightning implementation of ["SlowFast Networks for Video Recognition"](https://arxiv.org/abs/1812.03982).\
This impelmentation based on ["this repo"](https://github.com/r1ch88/SlowFastNetworks).

## Before start
Dataset should be orgnized as:  
```
dataset(e.g. UCF-101)  
│    │ train
│    │    │ ApplyEyeMakeup  
│    │    │ ApplyLipstick  
│    │    │ ...  
│    │ validation  
     │    │ ApplyEyeMakeup  
     │    │ ApplyLipstick  
     │    │ ...   
```
You can use `script/dir_sorting.py` to make the dataset like above.

## Usage
1. Train
```
python script/train.py --batch_size=32 --num_workers=30 --clip_len=64 --epochs=100
```
2. Tensorbaord
```
tensorboard --logdir=./log
```

## Requirements
python3\
PyTorch\
PyTorch-Lightning\
OpenCV  

## Code Reference:
[1] https://github.com/Guocode/SlowFast-Networks/  
[2] https://github.com/jfzhang95/pytorch-video-recognition  
[3] https://github.com/irhumshafkat/R2Plus1D-PyTorch
[4] https://github.com/r1ch88/SlowFastNetworks
