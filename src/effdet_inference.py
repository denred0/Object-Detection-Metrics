import sys

sys.path.insert(0, "model/effdet/EfficientDet")
from model import Inference  # import from github project, which added by sys
import os
from glob import glob
import random
import pandas as pd
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from ensemble_boxes import weighted_boxes_fusion
from tqdm import tqdm
from skimage import exposure

from pathlib import Path

SEED = 42


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


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


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT_PATH = 'data/effdet_inference/images'
# img_size_init = 1024
img_size = 768


class DatasetRetriever(Dataset):
    def __init__(self, images_names, transforms=None):
        super().__init__()
        self.image_names = images_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_name = self.image_names[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{image_name}.png', cv2.IMREAD_COLOR)
        # print('index', index)
        # print('path', f'{DATA_ROOT_PATH}/{image_name}.png')
        # print('image.shape', image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']
        return image, image_name

    def __len__(self) -> int:
        return self.image_names.shape[0]


def get_valid_transforms():
    return A.Compose([
        A.Resize(height=img_size, width=img_size, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


def collate_fn(batch):
    return tuple(zip(*batch))


def run_wbf(predictions, image_index, image_size=img_size, iou_thr=0.45, skip_box_thr=0.35, weights=None):
    boxes = [(prediction[image_index]['boxes'] / (image_size)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]

    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size)
    return boxes, scores, labels


def format_prediction_string(boxes):
    pred_strings = []
    for j in boxes:
        pred_strings.append("{0} {1} {2} {3};".format(j[0], j[1], j[2], j[3]))
    return "".join(pred_strings)


def make_predictions(model, images, score_threshold=0.20):
    images = torch.stack(images).cuda().float()
    # images = images.cuda().float()
    # print()
    predictions = []
    with torch.no_grad():
        img_size = torch.tensor([images[0].shape[-2:]] * 2, dtype=torch.float).to("cuda:0")
        det = model(images, torch.tensor([1] * images.shape[0]).float().cuda(), img_size=img_size)
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:, :4]
            scores = det[i].detach().cpu().numpy()[:, 4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]


def main():
    dataset = DatasetRetriever(
        images_names=np.array([path.split('/')[-1][:-4] for path in glob(f'{DATA_ROOT_PATH}/*.png')]),
        transforms=get_valid_transforms()
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        collate_fn=collate_fn
    )

    model = Inference(conf=f'tf_efficientdet_d6',
                      ckpt='model/effdet/best-checkpoint-021epoch.bin',
                      img_size=img_size, num_classes=1)
    model.to(device)

    # skip_box_thr=0.29
    # iou_thr=0.48

    gt_txts = get_all_files_in_folder(Path('data/effdet_inference/gt'), ['*.txt'])

    for image_ids, images in tqdm(enumerate(data_loader)):
        predictions = make_predictions(model, images[0], score_threshold=0.2)
        boxes, scores, labels = run_wbf(predictions, image_index=0, iou_thr=0.45, skip_box_thr=0.3)
        print(images[1][0])
        print(labels)
        print(scores)
        print(boxes)
        img = cv2.imread(DATA_ROOT_PATH + '/' + str(images[1][0]) + '.png', cv2.IMREAD_COLOR)

        xray_eq = exposure.equalize_hist(img)
        img = ((xray_eq - xray_eq.min()) * (1 / (xray_eq.max() - xray_eq.min()) * 255)).astype('uint8')

        h, w = img.shape[:2]

        coef_h = h / img_size
        coef_w = w / img_size

        for box in boxes:
            x1 = int(box[0] * coef_w)
            y1 = int(box[1] * coef_h)
            x2 = int(box[2] * coef_w)
            y2 = int(box[3] * coef_h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # draw gt
        for txt in gt_txts:
            if txt.stem == str(images[1][0]):
                print(txt.stem)
                preds = open(str(txt), 'r').readlines()

                # print()
                # print(preds)

                for pred in preds:
                    if pred != '\n':
                        pred_arr = np.array(pred.split(' '))

                        boxes = np.round(yolo2voc(img.shape[0], img.shape[1], pred_arr[1:5]))
                        boxes = [int(x) for x in list(boxes)]

                        print()
                        print(txt.stem)
                        print(boxes)
                        cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 3)

        cv2.imwrite('data/effdet_inference/result/' + str(images[1][0]) + '.png', img)

        # boxes = (boxes * coef).astype(np.int32).clip(min=0, max=1023)
        print()


if __name__ == '__main__':
    main()
