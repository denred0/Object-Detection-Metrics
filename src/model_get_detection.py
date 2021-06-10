from tqdm import tqdm
from pathlib import Path
from lib.Evaluator import *
from utils import get_all_files_in_folder


def get_detections_yolo4(images_source_dir, images_ext, txt_result_dir, inference_image_size, LABELS_FILE, CONFIG_FILE,
                         WEIGHTS_FILE,
                         MODEL_CONF_THR=0.5, NMS_THR=0.5):
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
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=MODEL_CONF_THR, nms_threshold=NMS_THR)

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
