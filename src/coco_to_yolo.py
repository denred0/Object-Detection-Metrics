import json
import pandas as pd
import shutil
import os

from pathlib import Path
from tqdm import tqdm
from utils import get_all_files_in_folder


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

    data = pd.read_csv('data/coco_to_yolo/output/annotations.csv')

    images = list(data['image_name'])
    bboxes = list(data['bbox'])
    widths = list(data['width'])
    heights = list(data['height'])
    classes = list(data['class_id'])

    breaked = 0

    all_images = set(images)

    for image in tqdm(all_images):
        filename, file_extension = os.path.splitext(image)

        for im, bbox, width, height, cl in zip(images, bboxes, widths, heights, classes):
            if im == image:
                shutil.copy('data/coco_to_yolo/input/images/' + image, dirpath)

                with open(str(dirpath) + '/' + filename + '.txt', "a") as myfile:
                    bbox = bbox.replace('[', '')
                    bbox = bbox.replace(']', '')

                    bbox_val = bbox.split(', ')

                    x_center = ((int(bbox_val[2]) - int(bbox_val[0])) / 2 + int(bbox_val[0])) / width
                    y_center = ((int(bbox_val[3]) - int(bbox_val[1])) / 2 + int(bbox_val[1])) / height

                    w = (int(bbox_val[2]) - int(bbox_val[0])) / width
                    h = (int(bbox_val[3]) - int(bbox_val[1])) / height

                    if w > 0 and h > 0:
                        mystring = str(cl) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)
                        myfile.write(mystring)
                        myfile.write("\n")
                    else:
                        breaked += 1

    print(breaked)


if __name__ == '__main__':
    # coco_to_csv()
    csv_to_yolo()
