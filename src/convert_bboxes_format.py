import shutil

from pathlib import Path

import cv2

from utils import get_all_files_in_folder
from numpy import loadtxt
from tqdm import tqdm


def convert_bboxes_format(input_txt_dir, output_txt_dir, input_format, output_format, one_class=None,
                          img_ext='png',
                          txt_ext=None,
                          delimiter_source_txt=' '):
    if txt_ext is None:
        txt_ext = ['*.txt']

    # clear folder
    dirpath = Path(output_txt_dir)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    txt_list = get_all_files_in_folder(input_txt_dir, txt_ext)

    for ind, txt in tqdm(enumerate(txt_list), total=len(txt_list)):
        image = cv2.imread(str(input_txt_dir) + '/' + txt.stem + '.' + img_ext, cv2.IMREAD_UNCHANGED)

        h, w = image.shape[:2]

        if input_format == 'yolo':
            if output_format == 'cxywh':
                filename = txt.stem
                lines = loadtxt(str(Path(input_txt_dir).joinpath(txt.name)), delimiter=delimiter_source_txt,
                                unpack=False)

                if lines.shape.__len__() == 1:
                    lines = [lines]

                with open(Path(output_txt_dir).joinpath(txt.name), 'w') as f:
                    for item in lines:
                        # print(filename)
                        # if filename == '01c3b410363f417836272fbc95eea13d4693ce18b653bc7219dc9433ce87fb91':
                        #     print()
                        # print(item)
                        # print(item[3])
                        # print(image_size_wh[0])
                        width = int(item[3] * w)
                        height = abs(int(item[4] * h))

                        if width > w: width = w
                        if height > h: height = h

                        x = abs(int(item[1] * w - width / 2))
                        y = abs(int(item[2] * h - height / 2))

                        if one_class:
                            label = one_class
                        else:
                            if item[0].is_integer():
                                label = int(item[0])
                            else:
                                label = item[0]

                        rec = str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height)
                        f.write("%s\n" % rec)

            elif output_format == 'cx1y1x2y2':
                filename = txt.stem
                lines = loadtxt(str(Path(input_txt_dir).joinpath(txt.name)), delimiter=delimiter_source_txt,
                                unpack=False)

                if lines.shape.__len__() == 1:
                    lines = [lines]

                with open(Path(output_txt_dir).joinpath(txt.name), 'w') as f:
                    if len(lines[0] != 0):
                        for item in lines:
                            # print(filename)
                            # if filename == '01c3b410363f417836272fbc95eea13d4693ce18b653bc7219dc9433ce87fb91':
                            #     print()
                            # print(item)
                            # print(item[3])
                            # print(image_size_wh[0])
                            width = int(item[3] * w)
                            height = abs(int(item[4] * h))

                            if width > w: width = w
                            if height > h: height = h

                            x1 = abs(int(item[1] * w - width / 2))
                            y1 = abs(int(item[2] * h - height / 2))

                            x2 = abs(int(item[1] * w + width / 2))
                            y2 = abs(int(item[2] * h + height / 2))

                            if one_class:
                                label = one_class
                            else:
                                if item[0].is_integer():
                                    label = int(item[0])
                                else:
                                    label = item[0]

                            rec = str(label) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2)
                            f.write("%s\n" % rec)


        elif input_format == 'cx1y1x2y2':
            if output_format == 'yolo':
                filename = txt.stem
                lines = loadtxt(str(Path(input_txt_dir).joinpath(txt.name)), delimiter=delimiter_source_txt,
                                unpack=False)

                if lines.shape.__len__() == 1:
                    lines = [lines]

                with open(Path(output_txt_dir).joinpath(txt.name), 'w') as f:
                    if len(lines[0] != 0):
                        for item in lines:
                            # print(filename)
                            # if filename == '01c3b410363f417836272fbc95eea13d4693ce18b653bc7219dc9433ce87fb91':
                            #     print()
                            # print(item)
                            # print(item[3])
                            # print(image_size_wh[0])
                            width = (int(item[3]) - int(item[1])) / w
                            height = (int(item[4]) - int(item[2])) / h

                            if width > 1: width = 1
                            if height > 1: height = 1

                            xcenter = ((int(item[3]) - int(item[1])) / 2 + int(item[1])) / w
                            ycenter = ((int(item[4]) - int(item[2])) / 2 + int(item[2])) / h

                            if one_class:
                                label = one_class
                            else:
                                if item[0].is_integer():
                                    label = int(item[0])
                                else:
                                    label = item[0]

                            rec = str(label) + ' ' + str(xcenter) + ' ' + str(ycenter) + ' ' + str(width) + ' ' + str(height)
                            f.write("%s\n" % rec)

        cv2.imwrite(str(output_txt_dir) + '/' + txt.stem + '.' + img_ext, image)


if __name__ == '__main__':
    input_txt_dir = Path('data/convert_bboxes_format/input_txt')
    output_txt_dir = Path('data/convert_bboxes_format/output_txt')
    input_format = 'yolo'  # class,  x_center, y_center, width, height, relative values
    # output_format = 'cxywh'  # class, x_left, y_top, width, height, absolute values
    output_format = 'cx1y1x2y2'  # class, x_left, y_top, x_right, y_bottom, absolute values
    # image_size_wh = [1920, 1080]
    img_ext = 'jpg'

    txt_ext = ['*.txt']

    convert_bboxes_format(input_txt_dir=input_txt_dir, output_txt_dir=output_txt_dir, input_format=input_format,
                          output_format=output_format, img_ext=img_ext, txt_ext=txt_ext)
