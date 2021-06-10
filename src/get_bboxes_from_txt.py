from lib.Evaluator import *
from utils import get_all_files_in_folder


def getBoundingBoxes(txt_dir, image_size, groundtruth=True):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()

    # Read ground truths
    files = get_all_files_in_folder(txt_dir, ['*.txt'])
    # Class representing bounding boxes (ground truths and detections)
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.stem
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            if groundtruth:
                idClass = splitLine[0]  # class
                x = float(splitLine[1])
                y = float(splitLine[2])
                w = float(splitLine[3])
                h = float(splitLine[4])
                bb = BoundingBox(nameOfImage, idClass, x, y, w, h, CoordinatesType.Absolute, (image_size[0], image_size[1]),
                                 BBType.GroundTruth,
                                 format=BBFormat.XYWH)
                allBoundingBoxes.addBoundingBox(bb)
            else:
                idClass = splitLine[0]  # class
                confidence = float(splitLine[1])  # confidence
                x = float(splitLine[2])
                y = float(splitLine[3])
                w = float(splitLine[4])
                h = float(splitLine[5])
                bb = BoundingBox(nameOfImage, idClass, x, y, w, h, CoordinatesType.Absolute, (image_size[0], image_size[1]),
                                 BBType.Detected,
                                 confidence, format=BBFormat.XYWH)
                allBoundingBoxes.addBoundingBox(bb)
        fh1.close()

    # # Read detections
    # files = get_all_files_in_folder(detections_dir, ['*.txt'])
    # # Read detections from txt file
    # # Each line of the files in the detections folder represents a detected bounding box.
    # # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # # Class_id represents the class of the detected bounding box
    # # Confidence represents the confidence (from 0 to 1) that this detection belongs to the class_id.
    # # x, y represents the most top-left coordinates of the bounding box
    # # x2, y2 represents the most bottom-right coordinates of the bounding box
    # for f in files:
    #     # nameOfImage = f.replace("_det.txt","")
    #     nameOfImage = f.stem
    #     # Read detections from txt file
    #     fh1 = open(f, "r")
    #     for line in fh1:
    #         line = line.replace("\n", "")
    #         if line.replace(' ', '') == '':
    #             continue
    #         splitLine = line.split(" ")
    #         idClass = splitLine[0]  # class
    #         confidence = float(splitLine[1])  # confidence
    #         x = float(splitLine[2])
    #         y = float(splitLine[3])
    #         w = float(splitLine[4])
    #         h = float(splitLine[5])
    #         bb = BoundingBox(nameOfImage, idClass, x, y, w, h, CoordinatesType.Absolute, (image_size, image_size),
    #                          BBType.Detected,
    #                          confidence, format=BBFormat.XYWH)
    #         allBoundingBoxes.addBoundingBox(bb)
    #     fh1.close()
    return allBoundingBoxes
