import numpy as np
import cv2
import glob
import os
from tqdm import tqdm

from pathlib import Path

from lib.BoundingBox import *
from lib.utils import *
from lib.BoundingBoxes import *
from lib.Evaluator import *


def get_all_files_in_folder(folder, types):
    files_grabbed = []
    for t in types:
        files_grabbed.extend(folder.rglob(t))
    files_grabbed = sorted(files_grabbed, key=lambda x: x)
    return files_grabbed


def get_detections_yolo4(images_source_dir, images_ext, txt_result_dir, inference_image_size, LABELS_FILE, CONFIG_FILE,
                         WEIGHTS_FILE,
                         MODEL_CONF_THR=0.5, NMS_THR=0.5,
                         NMS_SCORE_THR=0.5):
    np.random.seed(4)

    # get model
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    test_files = get_all_files_in_folder(images_source_dir, images_ext)

    for i, file in tqdm(enumerate(test_files)):

        image = cv2.imread(str(file), cv2.IMREAD_COLOR)

        (H, W) = image.shape[:2]
        # size = (W, H)
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (inference_image_size, inference_image_size), swapRB=True,
                                     crop=False)
        net.setInput(blob)

        layerOutputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > MODEL_CONF_THR:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=NMS_SCORE_THR, nms_threshold=NMS_THR)

        # ensure at least one detection exists
        txt_detection_result = []
        if len(idxs) > 0:
            box_string = ''
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (abs(boxes[i][0]), abs(boxes[i][1]))
                (w, h) = (boxes[i][2], boxes[i][3])

                conf = confidences[i]

                label = 0
                box_string = str(label) + ' ' + str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
                txt_detection_result.append(box_string)

            with open(Path(txt_result_dir).joinpath(file.stem + '.txt'), 'w') as f:
                for item in txt_detection_result:
                    f.write("%s\n" % item)

        else:
            with open(Path(txt_result_dir).joinpath(file.name), 'w') as f:
                f.write("%s\n" % '')


def getBoundingBoxes(groundtruths_dir, detections_dir, image_size):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()

    # Read ground truths
    files = get_all_files_in_folder(groundtruths_dir, ['*.txt'])
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
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(nameOfImage, idClass, x, y, w, h, CoordinatesType.Absolute, (image_size, image_size),
                             BBType.GroundTruth,
                             format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()

    # Read detections
    files = get_all_files_in_folder(detections_dir, ['*.txt'])
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents the confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.stem
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(nameOfImage, idClass, x, y, w, h, CoordinatesType.Absolute, (image_size, image_size),
                             BBType.Detected,
                             confidence, format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes


def evaluate(groundtruths_dir, detections_dir, image_size, images_source_dir, images_ext, images_draw_path,
             draw_images=False, IOUThreshold=0.3):
    # collect bboxes for evaluation
    allBoundingBoxes = getBoundingBoxes(groundtruths_dir=groundtruths_dir, detections_dir=detections_dir,
                                        image_size=image_size)

    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()

    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold)
    print("Metrics per class:\n")
    # Loop through classes to obtain their metrics
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('Class:', c)
        print('Precision:', round(np.mean(precision), 3))
        print('Recall:', round(np.mean(recall), 3))
        print('Average Precision (AP):', round(average_precision, 3))
        # print('%s: %f' % (c, average_precision))

    if draw_images:
        files = get_all_files_in_folder(images_source_dir, images_ext)

        for file in files:
            im = cv2.imread(str(file), cv2.IMREAD_COLOR)
            # Add bounding boxes
            im = allBoundingBoxes.drawAllBoundingBoxes(im, file.stem)
            cv2.imwrite(str(Path(images_draw_path).joinpath(file.name)), im)

        print("\nImages saved!")


def inference_and_create_txt_detections():
    # get inference + detection bboxes
    images_source_dir = Path('data/images/source')
    images_ext = ['*.jpg']
    txt_result_dir = Path('data/txt/detections')

    MODEL_CONF_THR = 0.5
    NMS_THR = 0.5
    NMS_SCORE_THR = 0.5

    inference_image_size = 1024

    LABELS_FILE = 'model/yolo4/obj.names'
    CONFIG_FILE = 'model/yolo4/yolov4-obj-mycustom.cfg'
    WEIGHTS_FILE = 'model/yolo4/yolov4-obj-mycustom_best.weights'

    get_detections_yolo4(images_source_dir=images_source_dir,
                         images_ext=images_ext,
                         txt_result_dir=txt_result_dir,
                         inference_image_size=inference_image_size,
                         LABELS_FILE=LABELS_FILE,
                         CONFIG_FILE=CONFIG_FILE,
                         WEIGHTS_FILE=WEIGHTS_FILE,
                         MODEL_CONF_THR=MODEL_CONF_THR,
                         NMS_THR=NMS_THR,
                         NMS_SCORE_THR=NMS_SCORE_THR)


def eval():
    # evaluation
    groundtruths_dir = Path('data/txt/groundtruths')
    detections_dir = Path('data/txt/detections')
    images_source_dir = Path('data/images/source')
    images_ext = ['*.jpg']
    images_draw_path = Path('data/images/images_result_bboxes')
    image_size = 1024
    evaluate(groundtruths_dir=groundtruths_dir,
             detections_dir=detections_dir,
             image_size=image_size,
             images_source_dir=images_source_dir,
             images_ext=images_ext,
             images_draw_path=images_draw_path,
             draw_images=True)


if __name__ == '__main__':
    # txt detections format:
    # class confidence x y width height
    # x y width height - absolute values

    # txt groundtruth format:
    # class x y width height
    # x y width height - absolute values

    # 1. Upload images for inference to data/images/source folder.
    # 2. Upload txt bboxes to data/txt/groundtruth folder. Files format see above.
    # 3. Configure model for prediction 1) add proper files in model/your_model folder and 2) corect inference_and_create_txt_detections() for your model.
    # 4. Execute inference_and_create_txt_detections(). As a result txt files in data/txt/detections folder will be created.
    # 5. Execute eval().


    # inference_and_create_txt_detections()
    eval()
