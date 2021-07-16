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

SEED = 42


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
        image_name = self.image_names[index].split('\\')[1]
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


def run_wbf(predictions, image_index, image_size=img_size, iou_thr=0.43, skip_box_thr=0.35, weights=None):
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
                      img_size=img_size)
    model.to(device)

    # skip_box_thr=0.29
    # iou_thr=0.48

    for image_ids, images in tqdm(enumerate(data_loader)):
        predictions = make_predictions(model, images[0])
        boxes, scores, labels = run_wbf(predictions, image_index=0)
        # boxes = (boxes * coef).astype(np.int32).clip(min=0, max=1023)


if __name__ == '__main__':
    main()
