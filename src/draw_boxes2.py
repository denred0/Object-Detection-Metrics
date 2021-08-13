import cv2
import shutil
import numpy as np

from pathlib import Path
from tqdm import tqdm
from numpy import loadtxt
import seaborn as sns

from utils import get_all_files_in_folder

labels_map = {0: 'pig',
              1: 'ladle',
              2: 'gates',
              3: 'person',
              4: 'red_pants',
              5: 'blue_pants'}

palette = sns.color_palette(palette='bright', n_colors=len(labels_map.values()))
palette_rgb = []
for p in palette:
    palette_rgb.append([int(np.clip(x * 255, 0, 255)) for x in p])


def draw_bbox(image, box, label, color):
    alpha = 0.1
    alpha_font = 0.6
    thickness = 4
    font_size = 1.0
    font_weight = 1
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_weight)[0]
    # cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
    #               color, -1)
    # cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    # cv2.rectangle(overlay_text, (box[0], box[1] - 18 - text_height), (box[0] + text_width + 8, box[1]),
    #               (0, 0, 0), -1)
    # cv2.addWeighted(overlay_text, alpha_font, output, 1 - alpha_font, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                  color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_weight, cv2.LINE_AA)
    return output


def plot_one_box(im, box, label=None, color=(0, 255, 255), line_thickness=2):
    c1 = (box[0], box[1])
    c2 = (box[2], box[3])

    tl = line_thickness or round(0.001 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    im = cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        im = cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        im = cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return im


def draw_bboxes(images_ext, label_types):
    output_dir = Path('data/draw_boxes2/output')
    if output_dir.exists() and output_dir.is_dir():
        shutil.rmtree(output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    images = get_all_files_in_folder(Path('data/draw_boxes2/input'), ['*.' + images_ext])
    txts = get_all_files_in_folder(Path('data/draw_boxes2/input'), ['*.txt'])

    for image_path, txt_path in tqdm(zip(images, txts), total=len(images)):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        height, width = image.shape[:2]

        if label_types == 'yolo':
            lines = loadtxt(str(txt_path), delimiter=' ', unpack=False)
            if lines.shape.__len__() == 1:
                lines = [lines]

            if len(lines[0] != 0):
                for line in lines:
                    # classId = line[0]
                    label = int(line[0])

                    # color = palette_rgb[label]
                    # label_str = labels_map[label]

                    xmin = int(float(line[1]) * width - float(line[3]) * width / 2)
                    ymin = int(float(line[2]) * height - float(line[4]) * height / 2)
                    xmax = int(float(line[1]) * width + float(line[3]) * width / 2)
                    ymax = int(float(line[2]) * height + float(line[4]) * height / 2)

                    box = [xmin, ymin, xmax, ymax]

                    image = plot_one_box(image, box, str(label), (0, 255, 0))

                    # if image_path.stem == '0':
                    #     cv2.rectangle(image, (94, 62), (333, 385), (255, 255, 255), 5)
                    #     print('height', height)
                    #     print('width', width)

                    # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)

        elif label_types == 'cx1y1x2y2':
            lines = loadtxt(str(txt_path), delimiter=' ', unpack=False)
            if lines.shape.__len__() == 1:
                lines = [lines]

            if len(lines[0] != 0):
                for line in lines:
                    label = int(line[0])

                    color = palette_rgb[label]
                    label_str = labels_map[label]

                    xmin = int(line[1])
                    ymin = int(line[2])
                    xmax = int(line[3])
                    ymax = int(line[4])
                    box = [xmin, ymin, xmax, ymax]
                    # image = draw_bbox(image, box, str(label), (0, 0, 255))

                    image = plot_one_box(image, box, str(label_str), color)

            cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)


if __name__ == '__main__':
    draw_bboxes('jpg', label_types='yolo')
