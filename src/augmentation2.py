import os
import cv2
import shutil

import albumentations as A

from tqdm import tqdm
from pathlib import Path


# def strong_aug(p=0.8):
#     return A.Compose([
#         # A.Rotate(limit=10, p=1.0),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         # A.Transpose(p=0.3),
#         # A.PadIfNeeded(min_height=300, min_width=300, always_apply=True, border_mode=0, p=1),
#         A.GaussNoise(p=0.4),
#         A.OneOf([
#             A.MotionBlur(p=0.1),
#             A.MedianBlur(blur_limit=3, p=0.1),
#             A.Blur(blur_limit=3, p=0.1),
#         ], p=0.2),
#         # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.8),
#         A.OneOf([
#             A.CLAHE(clip_limit=2),
#             A.Sharpen(),
#             A.Emboss(),
#             A.RandomBrightnessContrast(),
#         ], p=0.5),
#         A.HueSaturationValue(p=0.3),
#     ], p=p)


def strong_aug(p=0.8):
    return A.Compose([
        A.RandomSizedCrop(min_max_height=(800, 1024), height=1024, width=1024, p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                 val_shift_limit=0.2, p=0.9),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.9),
        ], p=0.9),
        A.ToGray(p=0.01),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.JpegCompression(quality_lower=85, quality_upper=95, p=0.1),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(p=1)], p=0.05),
        # A.Resize(height=img_size, width=img_size, p=1),
        A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
    ], p=1.0)


input_dir = "data/augmentation/images_txt_source/"
output_dir = "data/augmentation/images_txt_aug/"
files_list = sorted(list(os.walk(input_dir))[0][2])
images_list = [x for x in files_list if ".jpg" in x]


# delete output folder
dirpath = Path(output_dir)
if dirpath.exists() and dirpath.is_dir():
    shutil.rmtree(dirpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

addition_images_number = 2

for image in tqdm(images_list):

    augmentations = strong_aug(p=1.0)

    orig_image = cv2.imread("{}/{}".format(input_dir, image))

    with open("{}/{}.txt".format(input_dir, image.split(".")[0])) as l:

        lines = l.readlines()

        if len(lines) > 0:

            orig_label = lines

        else:

            orig_label = ""

    if orig_label == "":

        for idx in range(addition_images_number):
            data = {"image": orig_image}
            augmented = augmentations(**data)

            image_aug = augmented["image"]

            aug_image_path = output_dir + "{}_augm_{}.jpg".format(image.split(".")[0], idx)
            aug_txt_path = output_dir + "{}_augm_{}.txt".format(image.split(".")[0], idx)

            with open(aug_txt_path, "w") as l:
                l.write("")

            cv2.imwrite(aug_image_path, image_aug)

    else:

        class_idx = lines[0][0]
        bboxes = []

        for line in lines:
            orig_bbox = [float(x) for x in line[2:].split(" ")]
            bboxes.append(orig_bbox)

        bboxes_albumentations = A.augmentations.bbox_utils.convert_bboxes_to_albumentations(bboxes,
                                                                                            source_format="yolo",
                                                                                            rows=1024, cols=1024)

        for idx in range(addition_images_number):

            data = {"image": orig_image, "bboxes": bboxes_albumentations}
            augmented = augmentations(**data)

            image_aug = augmented["image"]
            bboxes_aug = augmented["bboxes"]

            bboxes_out = A.augmentations.bbox_utils.convert_bboxes_from_albumentations(bboxes_aug, "yolo", rows=1024,
                                                                                       cols=1024)

            aug_image_path = output_dir + "{}_augm_{}.jpg".format(image.split(".")[0], idx)
            aug_txt_path = output_dir + "{}_augm_{}.txt".format(image.split(".")[0], idx)

            with open(aug_txt_path, "w") as l:

                for bbox_out in bboxes_out:
                    l.write("{} {} {} {} {}\n".format(
                        class_idx,
                        bbox_out[0],
                        bbox_out[1],
                        bbox_out[2],
                        bbox_out[3],
                    ))

            cv2.imwrite(aug_image_path, image_aug)
