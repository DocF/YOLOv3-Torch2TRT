# -*- coding:utf-8 -*-
# Author: Richard Fang

import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorrt as trt
from torch2trt import torch2trt, tensorrt_converter


class UpsamlpleNet(nn.Module):
    def __init__(self):
        super(UpsamlpleNet, self).__init__()

    def forward(self, x):
        x = F.interpolate(x, size=(64, 64), mode='nearest')
        # print(x.type())
        # print(x)
        return x


if __name__ == '__main__':
    model = UpsamlpleNet().eval().cuda()
    data = torch.rand(size=(1, 3, 64, 64)).cuda()
    pred = model(data)

    #  TensorRT
    model_trt = torch2trt(model, [data])
    pred_trt = model_trt(data)

    # check the output against PyTorch
    print(torch.max(torch.abs(pred - pred_trt)))




