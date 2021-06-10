import shutil

from pathlib import Path
from utils import get_all_files_in_folder
from numpy import loadtxt
from tqdm import tqdm


def convert_bboxes_format(input_txt_dir, output_txt_dir, input_format, output_format, one_class=None,
                          image_size_wh=[1024, 1024],
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
                        width = int(item[3] * image_size_wh[0])
                        height = abs(int(item[4] * image_size_wh[1]))

                        if width > image_size_wh[0]: width = image_size_wh[0]
                        if height > image_size_wh[1]: width = image_size_wh[1]

                        x = abs(int(item[1] * image_size_wh[0] - width / 2))
                        y = abs(int(item[2] * image_size_wh[1] - height / 2))

                        if one_class:
                            label = one_class
                        else:
                            if item[0].is_integer():
                                label = int(item[0])
                            else:
                                label = item[0]

                        rec = str(label) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height)
                        f.write("%s\n" % rec)


if __name__ == '__main__':
    input_txt_dir = Path('data/convert_bboxes_format/input_txt')
    output_txt_dir = Path('data/convert_bboxes_format/output_txt')
    input_format = 'yolo'  # class,  x_center, y_center, width, height, relative values
    output_format = 'cxywh'  # class, x_left, y_top, width, height, absolute values
    image_size_wh = [1024, 1024]

    txt_ext = ['*.txt']

    convert_bboxes_format(input_txt_dir=input_txt_dir, output_txt_dir=output_txt_dir, input_format=input_format,
                          output_format=output_format, image_size_wh=image_size_wh, txt_ext=txt_ext)
