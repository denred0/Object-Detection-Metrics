import cv2
import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from ensemble_boxes import weighted_boxes_fusion, nms

import sys

sys.path.insert(0, "model/yolo5/yolov5-master")

from models.experimental import attempt_load
from utils.plots import colors, plot_one_box
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box


def inference_yolov5(input_path, img_size, conf_thres, iou_thres, weights):
    imgsz = img_size
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    max_det = 1000

    device = select_device('')
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    dataset = LoadImages(input_path, img_size=imgsz, stride=stride)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    results_output = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        img_id = path.split('/')[-1].split('.')[0]

        # Inference
        pred = model(img, augment=True, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        scores = []
        labels = []
        bboxes = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    conf_n = conf.detach().cpu().numpy().item()
                    if conf_n > conf_thres:
                        x1 = xyxy[0].detach().cpu().numpy().item()
                        y1 = xyxy[1].detach().cpu().numpy().item()
                        x2 = xyxy[2].detach().cpu().numpy().item()
                        y2 = xyxy[3].detach().cpu().numpy().item()

                        cl = int(cls)

                        scores.append(conf_n)
                        labels.append(cl)
                        bboxes.append([float(x1 / im0.shape[1]), float(y1 / im0.shape[0]), float(x2 / im0.shape[1]),
                                       float(y2 / im0.shape[0])])

                        # box_result = {}
                        # box_result['image_id'] = int(p.stem)
                        # box_result['category_id'] = cl
                        # box_result['score'] = conf_n
                        # box_result['bbox'] = [x1, y1, (x2 - x1), (y2 - y1)]

                        # results_output.append(box_result)
        results_output.append([int(img_id), scores, labels, bboxes, im0s.shape[1], im0s.shape[0]])

    return results_output


def run(input_path, output_path):
    img_size_model_1 = 640
    conf_thres_model_1 = 0.001
    iou_thres_model_1 = 0.5
    weights_model_1 = 'model/yolo5/weights1.pt'

    results_output_model_1 = inference_yolov5(input_path, img_size_model_1, conf_thres_model_1, iou_thres_model_1,
                                              weights_model_1)

    img_size_model_2 = 640
    conf_thres_model_2 = 0.001
    iou_thres_model_2 = 0.5
    weights_model_2 = 'model/yolo5/weights2.pt'

    results_output_model_2 = inference_yolov5(input_path, img_size_model_2, conf_thres_model_2, iou_thres_model_2,
                                              weights_model_2)

    ensemble_result = []
    weights = [3, 2]
    iou_thr = 0.5
    skip_box_thr = 0.001

    for m in tqdm(results_output_model_1):
        boxes = []
        scores = []
        labels = []

        w = m[4]
        h = m[5]

        scores.append(m[1])
        labels.append(m[2])
        boxes.append(m[3])

        for y in results_output_model_2:
            if m[0] == y[0]:
                boxes.append(y[3])
                scores.append(y[1])
                labels.append(y[2])
                break

        boxes_, scores_, labels_ = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                                         skip_box_thr=skip_box_thr)

        for box, score, label in zip(boxes_, scores_, labels_):
            box_result = {}
            box_result['image_id'] = int(m[0])
            box_result['category_id'] = int(label)
            box_result['score'] = float(score)
            box_result['bbox'] = [box[0] * w, box[1] * h, (box[2] - box[0]) * w, (box[3] - box[1]) * h]
            ensemble_result.append(box_result)

    with open(output_path, "w") as fp:
        fp.write(json.dumps(ensemble_result))

    print('\n' + json.dumps(ensemble_result))


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    # check_boxes()
    output_path = 'predict_yolo_ensemble.json'
    input_path = 'data/yolo5_inference/images'
    run(input_path, output_path)
