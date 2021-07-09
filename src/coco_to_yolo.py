import json

import cv2
import pandas as pd
import shutil
import os
import numpy as np

from pathlib import Path
from tqdm import tqdm
from utils import get_all_files_in_folder

from transliterate import translit, get_available_language_codes


def coco_to_csv():
    with open("data/coco_to_yolo/input/annotations.json") as f:
        data_annotation = json.load(f)

    images_list = get_all_files_in_folder(Path('data/coco_to_yolo/input/images'), ['*.jpg'])

    classes = data_annotation['categories']
    images = data_annotation['images']
    annotations = data_annotation['annotations']

    image_id = []
    image_name = []
    class_id = []
    class_name = []
    bbox = []
    area = []
    width = []
    height = []

    for image in tqdm(images):
        for annot in annotations:
            if annot['image_id'] == image['id']:
                image_id.append(image['id'])
                image_name.append(image['file_name'])
                width.append(image['width'])
                height.append(image['height'])

                class_id.append(annot['category_id'])
                area.append(annot['area'])
                bbox.append(annot['bbox'])

                for cl in classes:
                    if cl['id'] == annot['category_id']:
                        class_name.append(cl['name'])

    mydf = pd.DataFrame(list(zip(image_id, image_name, class_id, class_name, bbox, area, width, height)),
                        columns=['image_id', 'image_name', 'class_id', 'class_name', 'bbox', 'area', 'width', 'height'])

    mydf.to_csv('data/coco_to_yolo/output/annotations.csv')


def csv_to_yolo():
    dirpath = Path('data/coco_to_yolo/output/images_txt_yolo')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = Path('data/coco_to_yolo/output/images_draw_bboxes')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    dirpath = Path('data/coco_to_yolo/output/images_resized')
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv('data/coco_to_yolo/output/annotations.csv')

    images = list(data['image_name'])
    bboxes = list(data['bbox'])
    widths = list(data['width'])
    heights = list(data['height'])
    classes = list(data['class_id'])
    class_names = list(data['class_name'])

    classes_dict = {i: 0 for i in class_names}
    samples_per_class_limit = 20

    breaked = 0

    all_images = set(images)

    for image in tqdm(all_images):
        filename, file_extension = os.path.splitext(image)

        image_draw = cv2.imread('data/coco_to_yolo/input/images/' + image, cv2.IMREAD_COLOR)

        bboxes_total = []
        image_classes = []
        for im, bbox, width, height, cl, cl_name in zip(images, bboxes, widths, heights, classes, class_names):
            if im == image:

                bbox = bbox.replace('[', '')
                bbox = bbox.replace(']', '')

                bbox_val = bbox.split(', ')

                x1 = int(bbox_val[0])
                y1 = int(bbox_val[1])
                w = int(bbox_val[2])
                h = int(bbox_val[3])

                color = list(np.random.random(size=3) * 256)
                cv2.putText(image_draw, translit(cl_name, 'ru', reversed=True), (x1 + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                            cv2.LINE_AA)
                cv2.rectangle(image_draw, (x1, y1), (x1 + w, y1 + h), color, 2)

                x_center = (x1 + w / 2) / width
                y_center = (y1 + h / 2) / height

                wy = w / width
                hy = h / height

                bboxes_total.append([cl, x_center, y_center, wy, hy])


                # if x2 > x1 and y2 > y1:
                #     color = list(np.random.random(size=3) * 256)
                #     cv2.rectangle(image_draw, (x1, y1), (x2, y2), color, 2)
                #     cv2.putText(image_draw, translit(cl_name, 'ru', reversed=True), (x1 + 5, y1 + 25),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                #                 cv2.LINE_AA)
                #
                #     x_center = ((x2 - x1) / 2 + x1) / width
                #     y_center = ((y2 - y1) / 2 + y1) / height
                #
                #     w = (x2 - x1) / width
                #     h = (y2 - y1) / height
                #
                #     bboxes_total.append([cl, x_center, y_center, w, h])
                #     image_classes.append(cl_name)
                #     classes_dict[cl_name] += 1
                #
                # elif x1 > x2 and y1 > y2:
                #     color = list(np.random.random(size=3) * 256)
                #     cv2.rectangle(image_draw, (x2, y2), (x1, y1), color, 2)
                #     cv2.putText(image_draw, translit(cl_name, 'ru', reversed=True), (x2 + 5, y2 + 25),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                #                 cv2.LINE_AA)
                #
                #     x_center = ((x1 - x2) / 2 + x2) / width
                #     y_center = ((y2 - y1) / 2 + y2) / height
                #
                #     w = (x1 - x2) / width
                #     h = (y1 - y2) / height
                #
                #     bboxes_total.append([cl, x_center, y_center, w, h])
                #     image_classes.append(cl_name)
                #
                #     classes_dict[cl_name] += 1
                # else:
                #     color = list(np.random.random(size=3) * 256)
                #     cv2.putText(image_draw, str(bbox), (int(width / 2), int(height / 2)),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2,
                #                 cv2.LINE_AA)
                #     breaked += 1

        if len(bboxes_total) != 0:
            # limit_exceed = 0
            # for cl in image_classes:
            #     if classes_dict[cl] > samples_per_class_limit:
            #         limit_exceed += 1
            # if limit_exceed < len(image_classes):
            shutil.copy('data/coco_to_yolo/input/images/' + image, 'data/coco_to_yolo/output/images_txt_yolo')
            with open('data/coco_to_yolo/output/images_txt_yolo' + '/' + filename + '.txt', "a") as myfile:
                for box in bboxes_total:
                    mystring = str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(
                        box[4])
                    myfile.write(mystring)
                    myfile.write("\n")

        cv2.imwrite('data/coco_to_yolo/output/images_draw_bboxes' + '/' + image, image_draw)

    print(breaked)


def dataset_analyze():
    data_df = pd.read_csv('data/coco_to_yolo/output/annotations.csv')
    new_df = data_df[['image_id', 'class_id']].copy().drop_duplicates()
    print(new_df.groupby('class_id').count())


if __name__ == '__main__':
    # coco_to_csv()
    csv_to_yolo()
    # dataset_analyze()
