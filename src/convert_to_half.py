import cv2
import torch
import json
import os
from pathlib import Path

import sys

sys.path.insert(0, "model/yolo5/yolov5-master")

from models.experimental import attempt_load
from utils.plots import colors, plot_one_box
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box


def run(imgsz=640):
    device = select_device('0')
    model = attempt_load('model/yolo5/yolov5l6_f1_512.pt', map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.half()
    torch.save({'model': model}, 'model/yolo5/yolov5l6_f1_512_half.pt')


if __name__ == '__main__':
    # weights_path = sys.argv[1]
    run()
