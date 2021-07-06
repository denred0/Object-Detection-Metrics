import cv2
import shutil

from pathlib import Path
from tqdm import tqdm
from numpy import loadtxt

from utils import get_all_files_in_folder

output_dir = Path('data/draw_boxes2/output')
if output_dir.exists() and output_dir.is_dir():
    shutil.rmtree(output_dir)
Path(output_dir).mkdir(parents=True, exist_ok=True)

images = get_all_files_in_folder(Path('data/draw_boxes2/input'), ['*.jpg'])
txts = get_all_files_in_folder(Path('data/draw_boxes2/input'), ['*.txt'])

for image_path, txt_path in tqdm(zip(images, txts), total=len(images)):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    height, width = image.shape[:2]

    lines = loadtxt(str(txt_path), delimiter=' ', unpack=False).tolist()
    if not isinstance(lines[0], list):
        lines = [lines]

    for line in lines:
        classId = line[0]

        xmin = int(float(line[1]) * width - float(line[3]) * width / 2)
        ymin = int(float(line[2]) * height - float(line[4]) * width / 2)
        xmax = int(float(line[1]) * width + float(line[3]) * width / 2)
        ymax = int(float(line[2]) * height + float(line[4]) * width / 2)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 5)

    cv2.imwrite(str(output_dir.joinpath(image_path.name)), image)
