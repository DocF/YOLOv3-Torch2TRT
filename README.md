# YOLOv3-Torch2TRT

## Introduction
Convert YOLOv3 and YOLOv3-tiny (PyTorch version) into TensorRT models, through the torch2trt Python API.

## Installation 
#### Clone the repo
    git clone https://github.com/DocF/YOLOv3-Torch2TRT.git
    
#### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh
 
## Requirements
Two special Python packages are needed:
  
* tensorrt
  
* torch2trt
  
 Due to the upsampling operation in YOLO, according to torch2trt API introduction, you need to install the version with plugins.
 
 Installation reference: https://github.com/NVIDIA-AI-IOT/torch2trt
 
#### Check torch2trt API

    python3 check.py
 
 
## Inference Acceleration
Acceleration Techs：
* FP16
* TensorRT


Here are some results on TITAN xp:

| Model name | Input Size |  FP16 | Entire Mode*(FPS) | Backbone+FeatureNet(FPS) | 
|:---------: |------------|:-----:|:-----------------:|:-------------:|
| YOLOv3  | 320×320 |  | 87.58 Hz| 102.95 Hz| 
|         | 320×320 | ✔️ | 83.63 Hz| 100.36 Hz| 
| YOLOv3-TRT  | 320×320 |  | 110.74 Hz| 121.81 Hz| 
|             | 320×320 |  ✔️ | 106.92 Hz| 124.95 Hz| 
| YOLOv3-tiny  | 320×320 | | 354.10 Hz| 668.71 Hz| 
|              | 320×320 |  ✔️ | 379.11 Hz| 727.82 Hz| 
| YOLOv3-tiny-TRT | 320×320 |  |684.75 Hz| 1035.11 Hz| 
|                 | 320×320 |  ✔️ |649.71 Hz| 1012.66 Hz| 

Entire Model* = Backbone + Feature Net + YOLO Head

    python3 detect.py

## Statement
This repo is based on [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). Thx for the great repo.




