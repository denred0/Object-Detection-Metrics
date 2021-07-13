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
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box


def run(input_path, output_path):
    imgsz = 640
    conf_thres = 0.28
    iou_thres = 0.5
    classes = None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms = False
    max_det = 1000
    # line_thickness = 3
    save_dir = Path('data/yolo5_inference/result_images')
    save_crop = False
    # save_txt = False
    # save_img = False
    # view_img = False
    # hide_labels = False
    # save_conf = False
    # hide_conf = False

    device = select_device('')
    model = attempt_load('model/yolo5/best.pt', map_location=device)
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

        # Inference
        # t1 = time_synchronized()
        pred = model(img, augment=True, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            im_draw = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    conf_n = conf.numpy().item()
                    if conf_n > conf_thres:
                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        x1 = xyxy[0].numpy().item()
                        y1 = xyxy[1].numpy().item()
                        x2 = xyxy[2].numpy().item()
                        y2 = xyxy[3].numpy().item()

                        cv2.rectangle(im_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                        cl = int(cls)
                        cv2.putText(im_draw, str(cl), (int(x1) + 10, int(y1) + 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 255), 2,
                                    cv2.LINE_AA)
                        cv2.putText(im_draw, str(round(conf_n, 2)), (int(x1) + 10, int(y1) + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 255),
                                    2,
                                    cv2.LINE_AA)

                        box_result = {}
                        box_result['image_id'] = int(p.stem)
                        box_result['category_id'] = cl
                        box_result['score'] = conf_n
                        box_result['bbox'] = [x1, y1, (x2 - x1), (y2 - y1)]

                        results_output.append(box_result)

                        # if save_img or save_crop or view_img:  # Add bbox to image
                        #     c = int(cls)  # integer class
                        #     label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        #     plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        #     if save_crop:
                        #         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        filename = str(path).split('\\')[-1]
        cv2.imwrite(str(save_dir) + '/' + filename, im_draw)

        # dirname = os.path.dirname(output_path)
        # if not os.path.exists(dirname):
        #     os.makedirs(dirname)

    with open(output_path, "w") as fp:
        fp.write(json.dumps(results_output))

    print('\n' + json.dumps(results_output))
    # Print time (inference + NMS)
    # print(f'{s}Done. ({t2 - t1:.3f}s)')

def check_boxes():
    img = cv2.imread('data/yolo5_inference/images/0.jpg', cv2.IMREAD_COLOR)

    cv2.rectangle(img, (98, 61), (228, 324), (255, 0, 255), 1)

    cv2.rectangle(img, (98, 44), (325, 400), (0, 255, 0), 1)

    cv2.imwrite('data/yolo5_inference/images/0_0.jpg', img)

if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]

    check_boxes()
    # output_path = 'predict_yolo.json'
    # input_path = 'data/yolo5_inference/images'
    # run(input_path, output_path)
