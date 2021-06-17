import albumentations as A
import cv2
import os
from pathlib import Path
from numpy import loadtxt

from tqdm import tqdm

from utils import get_all_files_in_folder

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    # A.Rotate(limit=35, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
        A.MotionBlur(p=1.0)], p=0.5),
], bbox_params=A.BboxParams(format='yolo', min_visibility=0, label_fields=['class_labels']), p=1.0)


def create_augmentations(data_source_dir, label_type, data_aug_dir, images_ext, image_size):
    images = get_all_files_in_folder(data_source_dir, images_ext)

    for image_path in tqdm(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        filename = image_path.stem
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # if filename == '1a27d60cc0d7729fff321a53862908b6bfae1953b47b726386db7c2e92fd9d10':
        #     print()
        bboxes = []
        class_labels = []
        lines = loadtxt(str(os.path.join(image_path.parent, filename + '.txt')), delimiter=' ', unpack=False)
        if lines.shape.__len__() == 1:
            lines = [lines]
        for line in lines:
            if label_type == 'yolo':

                decrease_value = 0.999  # because albumentation generates bbox values grater than 1

                raw = [line[1] * decrease_value, line[2] * decrease_value,
                       line[3] * decrease_value, line[4] * decrease_value]
                bboxes.append(raw)

                if line[0].is_integer():
                    label = int(line[0])
                else:
                    label = line[0]
                class_labels.append(label)

        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # if len(transformed_bboxes) == 1:
        #     transformed_bboxes = [transformed_bboxes]

        with open(os.path.join(data_aug_dir, 'aug_' + image_path.stem + '.txt'), 'w') as f:
            for i, bbox in enumerate(transformed_bboxes):
                if label_type == 'yolo':
                    label = transformed_class_labels[i]

                    raw = str(label) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
                    f.write("%s\n" % raw)

        cv2.imwrite(os.path.join(data_aug_dir, 'aug_' + image_path.name), transformed_image)


if __name__ == '__main__':
    data_source_dir = Path('data/augmentation/images_txt_source')
    data_aug_dir = Path('data/augmentation/images_txt_aug')
    images_ext = ['*.jpg']
    label_type = 'yolo'
    image_size = [1024, 1024]

    create_augmentations(data_source_dir=data_source_dir,
                         label_type=label_type,
                         data_aug_dir=data_aug_dir,
                         images_ext=images_ext,
                         image_size=image_size)
