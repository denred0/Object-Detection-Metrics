import os
import sys
import cv2
import shutil
import numpy as np

from tqdm import tqdm
from pathlib import Path
from utils import get_all_files_in_folder, yolo2voc

# --------------------------------------------------------------------------------------------------
# Darknet initialization
# --------------------------------------------------------------------------------------------------
# sys.path.append('/home/vid/hdd/projects/darknet/')

from my_darknet import load_network, detect_image


dirpath = Path('data/yolo4_inference/result')
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

config_path = "data/yolo4_inference/cfg/yolov4-obj-mycustom.cfg"
weight_path = "data/yolo4_inference/cfg/yolov4-obj-mycustom_last.weights"
meta_path = "data/yolo4_inference/cfg/obj.data"

threshold = .4
hier_thresh = .5
nms_coeff = .45

net_main, class_names, colors = load_network(config_path, meta_path, weight_path)

images = get_all_files_in_folder(Path('data/yolo4_inference/images'), ['*.png'])

gt_txts = get_all_files_in_folder(Path('data/yolo4_inference/gt'), ['*.txt'])

results = []

for image in tqdm(images):

    imgArray = cv2.imread(str(image))
    imgArray_to_detect = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)

    detections = detect_image(net_main, class_names, imgArray_to_detect, thresh=threshold,
                              hier_thresh=hier_thresh, nms=nms_coeff)  # Class detection

    for detection in detections:

        if float(detection[1]) > threshold:

            current_class = detection[0]
            current_thresh = float(detection[1])
            current_coords = [float(x) for x in detection[2]]

            # print("Probability: {:.3f}, Class: {}".format(current_thresh, current_class))

            xmin = float(current_coords[0] - current_coords[2] / 2)
            ymin = float(current_coords[1] - current_coords[3] / 2)
            xmax = float(xmin + current_coords[2])
            ymax = float(ymin + current_coords[3])

            if xmin < 0:
                xmin = 0

            if xmax > imgArray.shape[1]:
                xmax = imgArray.shape[1]

            if ymin < 0:
                ymin = 0

            if ymax > imgArray.shape[0]:
                ymax = imgArray.shape[0]

            cv2.rectangle(imgArray, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), 2)

            # draw gt
            for txt in gt_txts:
                if txt.stem==image.stem:
                    preds = open(str(txt), 'r').readlines()

                    for pred in preds:
                        pred_arr = np.array(pred.split(' '))

                        boxes = np.round(yolo2voc(imgArray.shape[0], imgArray.shape[1], pred_arr[1:5]))
                        boxes = [int(x) for x in list(boxes)]

                        cv2.rectangle(imgArray, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)


            print(xmin, ymin, xmax, ymax)

    cv2.imwrite('data/yolo4_inference/result_images/' + image.name, imgArray)

print("Done!")
