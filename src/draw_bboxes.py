import shutil
import cv2

from pathlib import Path
from tqdm import tqdm

from utils import get_all_files_in_folder
from convert_bboxes_format import convert_bboxes_format
from get_bboxes_from_txt import getBoundingBoxes


def draw_bboxes(images_dir, txt_dir, output_images_dir, txt_format_dir, label_format, img_ext, txt_ext, image_size_wh):
    images = get_all_files_in_folder(images_dir, img_ext)
    txt = get_all_files_in_folder(txt_dir, txt_ext)

    if label_format == 'yolo':
        # clear folder
        dirpath = Path(txt_format_dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        convert_bboxes_format(input_txt_dir=txt_dir,
                              output_txt_dir=txt_format_dir,
                              input_format='yolo',
                              output_format='cxywh',
                              image_size_wh=image_size_wh)

        allBoundingBoxes = getBoundingBoxes(txt_dir=txt_format_dir,
                                            image_size=image_size_wh,
                                            groundtruth=True)
        # clear folder
        dirpath = Path(output_images_dir)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        files = get_all_files_in_folder(images_dir, img_ext)

        for ind, file in tqdm(enumerate(files), total=len(files)):
            im = cv2.imread(str(file), cv2.IMREAD_COLOR)
            # Add bounding boxes
            im = allBoundingBoxes.drawAllBoundingBoxes(im, file.stem)
            cv2.imwrite(str(Path(output_images_dir).joinpath(file.name)), im)

        print("\nImages saved!")


if __name__ == '__main__':
    images_dir = Path('data/draw_bboxes/images')
    txt_dir = Path('data/draw_bboxes/txt')
    output_images_dir = Path('data/draw_bboxes/images_result')
    txt_format_dir = Path('data/draw_bboxes/txt_convert')
    label_format = 'yolo'

    txt_ext = ['*.txt']
    img_ext = ['*.png']
    image_size_wh = [1024, 1024]

    draw_bboxes(images_dir=images_dir,
                txt_dir=txt_dir,
                output_images_dir=output_images_dir,
                txt_format_dir=txt_format_dir,
                label_format=label_format,
                img_ext=img_ext,
                txt_ext=txt_ext,
                image_size_wh=image_size_wh)
