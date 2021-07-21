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


def inference(source_images, config_path, weight_path, meta_path, threshold=0.2, hier_thresh=0.45, nms_coeff=0.5,
              images_ext='jpg'):
    dirpath = Path('data/yolo4_inference/result')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    net_main, class_names, colors = load_network(config_path, meta_path, weight_path)

    images = get_all_files_in_folder(Path(source_images), ['*.' + images_ext + ''])

    # gt_txts = get_all_files_in_folder(Path('data/yolo4_inference/images'), ['*.txt'])

    results = []

    thickness = 2

    for image in tqdm(images):

        imgArray = cv2.imread(str(image))
        imgArray_to_detect = cv2.cvtColor(imgArray, cv2.COLOR_BGR2RGB)

        h, w = imgArray_to_detect.shape[:2]
        detections = detect_image(net_main, class_names, imgArray_to_detect, thresh=threshold,
                                  hier_thresh=hier_thresh, nms=nms_coeff)  # Class detection

        detections_txt = []
        for detection in detections:

            if float(detection[1]) / 100 > threshold:

                current_class = detection[0]
                current_thresh = float(detection[1])
                current_coords = [float(x) for x in detection[2]]

                # print("Probability: {:.3f}, Class: {}".format(current_thresh, current_class))
                # format: x_center, y_center, w, h not normalized

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

                x_center_norm = float(current_coords[0]) / w
                y_center_norm = float(current_coords[1]) / h
                w_norm = float(current_coords[2]) / w
                h_norm = float(current_coords[3]) / h

                if w_norm > 1: w_norm = 1.0
                if h_norm > 1: h_norm = 1.0

                row = str(current_class) + ' ' + str(current_thresh / 100) + ' ' + str(x_center_norm) + ' ' + str(
                    y_center_norm) + ' ' + str(w_norm) + ' ' + str(h_norm)
                detections_txt.append(row)

                cv2.rectangle(imgArray, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 255), thickness)

        with open(Path(dirpath).joinpath(image.stem + '.txt'), 'w') as f:
            for item in detections_txt:
                f.write("%s\n" % item)

        # # draw gt
        # for txt in gt_txts:
        #     if txt.stem == image.stem:
        #         preds = open(str(txt), 'r').readlines()
        #
        #         # print()
        #         # print(preds)
        #
        #         for pred in preds:
        #             if pred != '\n':
        #                 pred_arr = np.array(pred.split(' '))
        #
        #                 boxes = np.round(yolo2voc(imgArray.shape[0], imgArray.shape[1], pred_arr[1:5]))
        #                 boxes = [int(x) for x in list(boxes)]
        #
        #                 cv2.rectangle(imgArray, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), thickness)

        # print(xmin, ymin, xmax, ymax)

        # cv2.imwrite('data/yolo4_inference/result_images/' + image.name, imgArray)

    print("Done!")


if __name__ == '__main__':
    # source_images = sys.argv[1]
    # config_path = sys.argv[2]
    # weight_path = sys.argv[3]
    # meta_path = sys.argv[4]
    # threshold = float(sys.argv[5])
    # hier_thresh = float(sys.argv[6])
    # nms_coeff = float(sys.argv[7])
    # images_ext = sys.argv[8]

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default='data/yolo4_inference/images', help='default %(default)s')
    parser.add_argument("--config", default='data/yolo4_inference/cfg/yolov4-obj-mycustom.cfg',
                        help='default %(default)s')
    parser.add_argument("--weights", default='data/yolo4_inference/cfg/yolov4-obj-mycustom_best.weights',
                        help='default %(default)s')
    parser.add_argument("--meta", default='data/yolo4_inference/cfg/obj.data', help='default %(default)s')
    parser.add_argument("--threshold", "--th", default=0.2, help='default %(default)s')
    parser.add_argument("--hier_thresh", '-ht', default=0.5, help='default %(default)s')
    parser.add_argument("--nms_coeff", '-nms', default=0.5, help='default %(default)s')
    parser.add_argument("--images_ext", '-ext', default='jpg', help='default %(default)s')

    args = parser.parse_args()

    # source_images = 'data/yolo4_inference/images'
    # config_path = "data/yolo4_inference/cfg/yolov4-obj-mycustom.cfg"
    # weight_path = "data/yolo4_inference/cfg/yolov4-obj-mycustom_best.weights"
    # meta_path = "data/yolo4_inference/cfg/obj.data"
    # threshold = 0.2
    # hier_thresh = 0.45
    # nms_coeff = 0.5
    # images_ext = 'jpg'

    inference(args.source, args.config, args.weights, args.meta, float(args.threshold),
              float(args.hier_thresh), float(args.nms_coeff), args.images_ext)
