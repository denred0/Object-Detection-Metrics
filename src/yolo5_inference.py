import os
import numpy as np
import time
import cv2
import sys
import json
import torch

import sys
sys.path.insert(0, "model/yolo5/yolov5-master")

from pathlib import Path
import glob
from tqdm import tqdm


# from ensemble_boxes import *


def run(input_path, output_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/yolo5/best.pt', force_reload=True)
    model.conf = 0.18  # confidence threshold (0-1)
    model.iou = 0.4  # NMS IoU threshold (0-1)
    model.classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]

    # test_folder = Path(input_path)
    # types = ('*.jpg')
    # COLOR = (255, 0, 0)

    results_output = []
    for i, image_path in enumerate(os.listdir(input_path)):
        img_path = input_path + '/' + str(image_path)
        # image_id, ext = os.path.splitext(img_path)
        image_id = int(img_path.split('/')[-1][:-4])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # img = torch.from_numpy(image.astype(np.float32))#.to(device)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_channels = image.shape
        # img /= 255.0
        # img = img.unsqueeze(0)

        results = model(image)
        if image_id == 13:
            print()

        results_list = results.pandas().xyxy[0].values.tolist()

        # # ensure at least one detection exists
        if len(results_list) > 0:
            for res in results_list:
                box_result = {}

                # extract the bounding box coordinates
                (x_min, y_min) = (res[0], res[1])
                (x_max, y_max) = (res[2], res[3])
                width = x_max - x_min
                height = y_max - y_min

                cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 255), 3)

                box_result['image_id'] = image_id
                box_result['category_id'] = res[5]
                box_result['score'] = res[4]
                box_result['bbox'] = [x_min, y_min, width, height]

                results_output.append(box_result)

        cv2.imwrite('data/yolo5_inference/result_images/' + str(img_path.split('/')[-1]), image)

    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(output_path, "w") as fp:
        fp.write(json.dumps(results_output))

    print(json.dumps(results_output))


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    input_path = 'data/yolo5_inference/images'
    output_path = 'data/yolo5_inference/predict_yolo.json'
    run(input_path, output_path)
