import os
import cv2
import numpy as np
import json
import sys

from pathlib import Path


def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y2]

    """
    bboxes = bboxes.copy().astype(float)  # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]] / 2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def run(input_path, output_path):
    images_dir = input_path
    weights_dir = 'model/yolo5/best.pt'
    conf = 0.1
    iou = 0.5
    img_size = 640

    os.system(
        f"python model/yolo5/yolov5-master/detect.py --weights {weights_dir} --img {img_size} --conf {conf} --iou {iou} "
        f"--source {images_dir} --save-txt --save-conf --exist-ok")

    all_txts = get_all_files_in_folder(Path('runs/detect/exp/labels'), ['*.txt'])
    all_images = get_all_files_in_folder(Path('runs/detect/exp'), ['*.jpg'])

    results_output = []
    for txt in all_txts:
        for img in all_images:
            if img.stem == txt.stem:

                image = cv2.imread(str(img), cv2.IMREAD_COLOR)
                h, w, _ = image.shape

                preds = open(str(txt), 'r').readlines()

                for pred in preds:
                    pred_arr = np.array(pred.split(' '))

                    boxes = np.round(yolo2voc(h, w, pred_arr[1:5]))
                    boxes = [int(x) for x in list(boxes)]

                    box_result = {}
                    box_result['image_id'] = int(img.stem)
                    box_result['category_id'] = int(pred_arr[0])
                    box_result['score'] = float(pred_arr[5])
                    box_result['bbox'] = [boxes[0], boxes[1], boxes[2] - boxes[0], boxes[3] - boxes[1]]

                    results_output.append(box_result)

    with open(output_path, "w") as fp:
        fp.write(json.dumps(results_output))

    print('\n' + json.dumps(results_output))


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    output_path = 'predict_yolo.json'
    input_path = 'data/yolo5_inference/images'
    run(input_path, output_path)
