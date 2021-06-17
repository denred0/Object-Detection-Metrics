import os
import cv2

import albumentations as A

from tqdm import tqdm


def strong_aug(p=0.8):
    return A.Compose([
        # A.Rotate(limit=10, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Transpose(p=0.3),
        # A.PadIfNeeded(min_height=300, min_width=300, always_apply=True, border_mode=0, p=1),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.4),
        A.OneOf([
            A.MotionBlur(p=0.1),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.8),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.5),
        A.HueSaturationValue(p=0.3),
    ], p=p)


images_dir = "img/"
files_list = sorted(list(os.walk(images_dir))[0][2])
images_list = [x for x in files_list if ".png" in x]

addition_images_number = 1

for image in tqdm(images_list):

    augmentations = strong_aug(p=1.0)

    orig_image = cv2.imread("{}/{}".format(images_dir, image))

    with open("{}/{}.txt".format(images_dir, image.split(".")[0])) as l:

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

            aug_image_path = "img_2/{}_augm_{}.png".format(image.split(".")[0], idx)
            aug_txt_path = "img_2/{}_augm_{}.txt".format(image.split(".")[0], idx)

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
                                                                                            rows=1520, cols=2688)

        for idx in range(addition_images_number):

            data = {"image": orig_image, "bboxes": bboxes_albumentations}
            augmented = augmentations(**data)

            image_aug = augmented["image"]
            bboxes_aug = augmented["bboxes"]

            bboxes_out = A.augmentations.bbox_utils.convert_bboxes_from_albumentations(bboxes_aug, "yolo", rows=1520,
                                                                                       cols=2688)

            aug_image_path = "img_2/{}_augm_{}.png".format(image.split(".")[0], idx)
            aug_txt_path = "img_2/{}_augm_{}.txt".format(image.split(".")[0], idx)

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
