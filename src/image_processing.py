import cv2
import shutil

from tqdm import tqdm
from pathlib import Path

from utils import get_all_files_in_folder


def png_to_jpg(input_dir, output_dir):
    # delete output folder
    dirpath = output_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    images_paths = get_all_files_in_folder(input_dir, ['*.jpg'])

    for image_path in tqdm(images_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        cv2.imwrite(str(output_dir.joinpath(image_path.stem + '.png')), image)




if __name__ == '__main__':
    input_dir = Path('data/png_to_jpg/input')
    output_dir = Path('data/png_to_jpg/output')
    png_to_jpg(input_dir, output_dir)
