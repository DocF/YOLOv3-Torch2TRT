from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from torchstat import stat
from torch2trt import torch2trt


def Speed(TensorRT, Half):
    if Half:
        test_data = torch.rand(size=(1000, 1, 3, opt.img_size, opt.img_size)).cuda().half()
    else:
        test_data = torch.rand(size=(1000, 1, 3, opt.img_size, opt.img_size)).cuda()

    print("Start test speed")
    # 模型推理
    if TensorRT:

        start = time.time()
        for i in range(test_data.size()[0]):
            detections = model_backbone(test_data[i])
        end = time.time()
        if Half is False:
            print("Fp32 Backbone Speed:", 1 / (end - start) * test_data.size()[0], "Hz")
        else:
            print("Fp16 Backbone Speed:", 1 / (end - start) * test_data.size()[0], "Hz")

        start = time.time()
        for i in range(test_data.size()[0]):
            detections = model_trt(test_data[i])
        end = time.time()
        if Half is False:
            print("Fp32 TensorRT Backbone Speed:", 1 / (end - start) * test_data.size()[0], "Hz")
        else:
            print("Fp16 TensorRT Backbone Speed:", 1 / (end - start) * test_data.size()[0], "Hz")

        start = time.time()
        for i in range(test_data.size()[0]):
            detections = yolo_head(model_trt(test_data[i]))
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
        end = time.time()
        if Half is False:
            print("Fp32 TensorRT Model Detect Speed:", 1 / (end - start) * test_data.size()[0], "Hz")
        else:
            print("Fp16 TensorRT Model Detect Speed:", 1 / (end - start) * test_data.size()[0], "Hz")

    else:

        start = time.time()
        for i in range(test_data.size()[0]):
            detections = model(test_data[i])
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
        end = time.time()
        if Half is False:
            print("Fp32 Original Model Detect Speed:", 1 / (end - start) * test_data.size()[0], "Hz")
        else:
            print("Fp16 Original Model Detect Speed:", 1 / (end - start) * test_data.size()[0], "Hz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples/", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # 模型选择
    # TensorRT 只能加速 YOLO 的特征提取网络, YOLO Head 目前还不能应用
    Speed_Test = True
    TensorRT = True
    Half = True    # 半精度

    if TensorRT is True:
        if Half is True:
            model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.img_size).to(device).half()
        else:
            model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.img_size).to(device)

        # 权重加载
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model_backbone.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model_backbone.load_state_dict(torch.load(opt.weights_path))
        # Set in evaluation mode 前向推理时候会忽略 BatchNormalization 和 Dropout
        model_backbone.eval()
        # 添加 Detection Head
        yolo_head = YOLOHead(config_path=opt.model_def)

        # DarknetBackbone 转换为 TensorRT 模型
        if Half is True:
            x = torch.rand(size=(1, 3,  opt.img_size, opt.img_size)).cuda().half()
            model_trt = torch2trt(model_backbone, [x], fp16_mode=True)
        else:
            x = torch.rand(size=(1, 3,  opt.img_size, opt.img_size)).cuda()
            model_trt = torch2trt(model_backbone, [x])

    else:
        if Half:
            model = Darknet(opt.model_def, img_size=opt.img_size, TensorRT=False, Half=True).to(device).half()
        else:
            model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

        # 权重加载
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))
        # Set in evaluation mode 前向推理时候会忽略 BatchNormalization 和 Dropout
        model.eval()

    # 速度测试
    if Speed_Test:
        Speed(TensorRT, Half)

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # # YOLO 参数、运算量测试
    # model.to("cpu")
    # stat(model, input_size=(3, 416, 416))

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # # 半精度模型输入
        # input_imgs = Variable(input_imgs.type(Tensor)).half()

        # Get detections
        with torch.no_grad():

            # 注释说明
            # YOLOv3  return tensor size [batch_size, 10647, 85]
            # YOLOv3-tiny return tensor size [batch_size, 2535, 85]
            # 10647 = 3×13×13 + 3×26×26 + 3×52×52
            # 2535 = 3×13×13 + 3×26×26
            # 85: 其中前4维度为坐标，第5个维度为bbox的置信度，后面80个维度为coco80类目标的对应概率

            # TensorRT 加速
            if TensorRT:
                detections = yolo_head(model_trt(input_imgs))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=2)
            else:
                detections = model(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=2)

        # # 保存模型
        # torch.save(model, "YOLOv3.pth")
        # # 打印模型
        # print(model)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
