import os
import shutil
import numpy as np

from pathlib import Path
from tqdm import tqdm

from utils import get_all_files_in_folder


def generate_train_test(data_dir, split):
    images = get_all_files_in_folder(data_dir, ['*.jpg'])

    with open('data/darknet_prepare_for_train/' + split + '.txt', "w") as outfile:
        for image in images:
            outfile.write('data/darknet_prepare_for_train/' + split + '/' + image.name)
            outfile.write("\n")
        outfile.close()


# darknet_path = Path('/home/vid/hdd/projects/darknet/my_data')

root_dir = Path('data/darknet_prepare_for_train/0_dataset')
root_data_jpg_dir = Path('data/darknet_prepare_for_train/data_jpg')
root_data_txt_dir = Path('data/darknet_prepare_for_train/data_txt')
train_dir = Path('data/darknet_prepare_for_train/train')
test_dir = Path('data/darknet_prepare_for_train/test')
backup_dir = Path('data/darknet_prepare_for_train/backup')

dirpath = root_data_jpg_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = root_data_txt_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = train_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = test_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

dirpath = backup_dir
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

all_images = get_all_files_in_folder(root_dir, ['*.jpg'])
all_txts = get_all_files_in_folder(root_dir, ['*.txt'])
print(f'Total images: {len(all_images)}')
print(f'Total labels: {len(all_txts)}')

for img in tqdm(all_images):
    shutil.copy(img, root_data_jpg_dir)

for txt in tqdm(all_txts):
    shutil.copy(txt, root_data_txt_dir)

val_part = 0.2
np.random.shuffle(all_images)
train_FileNames, val_FileNames = np.split(np.array(all_images), [int(len(all_images) * (1 - val_part))])

for name in tqdm(train_FileNames):
    shutil.copy(name, train_dir)
    shutil.copy(root_data_txt_dir.joinpath(name.stem + '.txt'), train_dir)

for name in tqdm(val_FileNames):
    shutil.copy(name, test_dir)
    shutil.copy(root_data_txt_dir.joinpath(name.stem + '.txt'), test_dir)

generate_train_test(train_dir, 'train')
generate_train_test(test_dir, 'test')

# copy cfg data
# shutil.copy('data/darknet_prepare_for_train/0_cfg/obj.data', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_cfg/obj.names', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_cfg/yolov4-obj-mycustom.cfg', darknet_path)
# shutil.copy('data/darknet_prepare_for_train/0_weights/yolov4-p5.conv.232', darknet_path)

os.system("/home/vid/hdd/projects/darknet/darknet detector train "
          "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_cfg/obj.data "
          "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_cfg/yolov4-obj-mycustom.cfg "
          "/home/vid/hdd/projects/PycharmProjects/Object-Detection-Metrics/data/darknet_prepare_for_train/0_weights/yolov4-p5.conv.232 -map")

# ./darknet detector train my_data/obj.data my_data/yolov4-obj-mycustom.cfg my_data/yolov4-p5.conv.232 -dont_show -map
