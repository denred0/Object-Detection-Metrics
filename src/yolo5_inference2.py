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


def run(input_path, output_path):
    imgsz = 640
    conf_thres = 0.1
    iou_thres = 0.2
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    max_det = 1000
    # save_dir = Path('data/yolo5_inference/result_images')
    # Path("not_found_images").mkdir(parents=True, exist_ok=True)

    device = select_device('')
    model = attempt_load('model/yolo5/yolov5l6_f0.pt', map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    dataset = LoadImages(input_path, img_size=imgsz, stride=stride)
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # counter = 0
    # not_found = []
    results_output = []
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=True, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # filename = str(path).split('/')[-1]

        paint = False
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # im_draw = im0.copy()

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

                        # cv2.rectangle(im_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                        cl = int(cls)
                        # cv2.putText(im_draw, str(cl), (int(x1) + 10, int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             (255, 0, 255), 2,
                        #             cv2.LINE_AA)
                        # cv2.putText(im_draw, str(round(conf_n, 2)), (int(x1) + 10, int(y1) + 70),
                        #             cv2.FONT_HERSHEY_SIMPLEX,
                        #             1, (255, 0, 255),
                        #             2,
                        #             cv2.LINE_AA)

                        # paint = True

                        box_result = {}
                        box_result['image_id'] = int(p.stem)
                        box_result['category_id'] = cl
                        box_result['score'] = conf_n
                        box_result['bbox'] = [x1, y1, (x2 - x1), (y2 - y1)]

                        results_output.append(box_result)

        # if not paint:
        #     cv2.imwrite('not_found_images' + '/' + filename, im0s)
        #     not_found.append(path)

        # cv2.imwrite(str(save_dir) + '/' + filename, im_draw)

    # conf_thres *= 0.7
    # if len(not_found) != 0:
    #     dataset_not_found = LoadImages('not_found_images', img_size=imgsz, stride=stride)
    #     for path, img, im0s, vid_cap in dataset_not_found:
    #         img = torch.from_numpy(img).to(device)
    #         img = img.float()
    #         img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #         if img.ndimension() == 3:
    #             img = img.unsqueeze(0)
    #
    #             # Inference
    #             pred = model(img, augment=True, visualize=False)[0]
    #
    #             # Apply NMS
    #             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    #
    #             filename = str(path).split('/')[-1]
    #
    #             paint = False
    #             # Process detections
    #             for i, det in enumerate(pred):  # detections per image
    #                 p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
    #
    #                 p = Path(p)  # to Path
    #                 im_draw = im0.copy()
    #
    #                 if len(det):
    #                     # Rescale boxes from img_size to im0 size
    #                     det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    #
    #                     # Write results
    #                     for *xyxy, conf, cls in reversed(det):
    #                         conf_n = conf.detach().cpu().numpy().item()
    #                         if conf_n > conf_thres:
    #                             x1 = xyxy[0].detach().cpu().numpy().item()
    #                             y1 = xyxy[1].detach().cpu().numpy().item()
    #                             x2 = xyxy[2].detach().cpu().numpy().item()
    #                             y2 = xyxy[3].detach().cpu().numpy().item()
    #
    #                             # cv2.rectangle(im_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
    #                             cl = int(cls)
    #                             # cv2.putText(im_draw, str(cl), (int(x1) + 10, int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                             #             (255, 0, 255), 2,
    #                             #             cv2.LINE_AA)
    #                             # cv2.putText(im_draw, str(round(conf_n, 2)), (int(x1) + 10, int(y1) + 70),
    #                             #             cv2.FONT_HERSHEY_SIMPLEX,
    #                             #             1, (255, 0, 255),
    #                             #             2,
    #                             #             cv2.LINE_AA)
    #
    #                             paint = True
    #
    #                             box_result = {}
    #                             box_result['image_id'] = int(p.stem)
    #                             box_result['category_id'] = cl
    #                             box_result['score'] = conf_n
    #                             box_result['bbox'] = [x1, y1, (x2 - x1), (y2 - y1)]
    #
    #                             results_output.append(box_result)

            # cv2.imwrite(str(save_dir) + '/' + filename, im_draw)

    # print('ddddd', counter)

    with open(output_path, "w") as fp:
        fp.write(json.dumps(results_output))

    print('\n' + json.dumps(results_output))


def check_boxes():
    img = cv2.imread('data/yolo5_inference/images/0.jpg', cv2.IMREAD_COLOR)

    cv2.rectangle(img, (98, 61), (228, 324), (255, 0, 255), 1)

    cv2.rectangle(img, (98, 44), (325, 400), (0, 255, 0), 1)

    cv2.imwrite('data/yolo5_inference/images/0_0.jpg', img)


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    # check_boxes()
    output_path = 'predict_yolo.json'
    input_path = 'data/yolo5_inference/images'
    run(input_path, output_path)
