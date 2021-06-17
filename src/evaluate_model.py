import shutil

from pathlib import Path
from lib.Evaluator import *
from utils import get_all_files_in_folder
from model_get_detection import get_detections_yolo4
from get_bboxes_from_txt import getBoundingBoxes


def inference_and_create_txt_detections(type):
    # clear folder
    txt_result_dir = Path('data/evaluate_model/txt/detections')
    dirpath = txt_result_dir
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    images_source_dir = Path('data/evaluate_model/images/source')
    images_ext = ['*.jpg']

    if type == 'yolo':
        # get inference + detection bboxes

        MODEL_CONF_THR = 0.25
        NMS_THR = 0.5

        inference_image_size = 736

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
                             NMS_THR=NMS_THR)


def eval():
    # evaluation
    groundtruths_dir = Path('data/evaluate_model/txt/groundtruths')
    detections_dir = Path('data/evaluate_model/txt/detections')
    images_source_dir = Path('data/evaluate_model/images/source')
    images_ext = ['*.jpg']
    images_draw_path = Path('data/evaluate_model/images/eval_draw_bboxes')
    image_size = [1024, 1024]
    evaluate(groundtruths_dir=groundtruths_dir,
             detections_dir=detections_dir,
             image_size=image_size,
             images_source_dir=images_source_dir,
             images_ext=images_ext,
             images_draw_path=images_draw_path,
             draw_images=True)


def evaluate(groundtruths_dir, detections_dir, image_size, images_source_dir, images_ext, images_draw_path,
             draw_images=False, IOUThreshold=0.3):
    # collect bboxes for evaluation

    allBoundingBoxes = BoundingBoxes()
    gtBoundingBoxes = getBoundingBoxes(txt_dir=groundtruths_dir, image_size=image_size, groundtruth=True)
    allBoundingBoxes._boundingBoxes.extend(gtBoundingBoxes._boundingBoxes)

    detBoundingBoxes = getBoundingBoxes(txt_dir=detections_dir, image_size=image_size, groundtruth=False)
    allBoundingBoxes._boundingBoxes.extend(detBoundingBoxes._boundingBoxes)

    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()

    metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold)
    print("Metrics per class:")
    # Loop through classes to obtain their metrics
    average_precision = 0
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        total_positivies = mc['total positives']
        total_tp = mc['total TP']
        total_fp = mc['total FP']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('Class:', c)
        print('Precision:', round(np.mean(precision), 4))
        print('Recall:', round(np.mean(recall), 4))
        print('FP rate:',  round((total_fp) / total_positivies, 4))
        print('Average Precision (AP):', round(average_precision, 4))
        print('total TP', total_tp)
        print('total FP', total_fp)
        print('interpolated precision', round(np.mean(ipre), 4))
        print('interpolated recall', round(np.mean(irec), 4))
        print()
        # print('%s: %f' % (c, average_precision))

    if draw_images:
        # recreate folder
        dirpath = Path(images_draw_path)
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        Path(dirpath).mkdir(parents=True, exist_ok=True)

        files = get_all_files_in_folder(images_source_dir, images_ext)

        for file in files:
            im = cv2.imread(str(file), cv2.IMREAD_COLOR)
            # Add bounding boxes
            im = allBoundingBoxes.drawAllBoundingBoxes(im, file.stem)
            cv2.imwrite(str(Path(images_draw_path).joinpath(file.name)), im)

        print("\nImages saved!")

    return average_precision


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

    # type = 'yolo'
    # inference_and_create_txt_detections(type=type)

    eval()
