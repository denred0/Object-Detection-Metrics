from sklearn.model_selection import ParameterGrid
from pathlib import Path
from evaluate_model import *
from datetime import datetime
import json


def find_best_detection_params():
    images_source_dir = Path('data/evaluate_model/images/source')
    images_ext = ['*.jpg']
    txt_result_dir = Path('data/evaluate_model/txt/detections')

    LABELS_FILE = 'model/yolo4/obj.names'
    CONFIG_FILE = 'model/yolo4/yolov4-obj-mycustom.cfg'
    WEIGHTS_FILE = 'model/yolo4/yolov4-obj-mycustom_best.weights'

    groundtruths_dir = Path('data/evaluate_model/txt/groundtruths')
    detections_dir = Path('data/evaluate_model/txt/detections')
    images_source_dir = Path('data/evaluate_model/images/source')
    images_ext = ['*.jpg']
    images_draw_path = Path('data/evaluate_model/images/eval_draw_bboxes')
    image_size = [1024, 1024]

    param_grid = {'MODEL_CONF_THR': [0.2, 0.22, 0.24, 0.26, 0.28],
                  'NMS_THR': [0.5, 0.52, 0.54, 0.56, 0.58],
                  'inference_image_size': [704, 736, 768]}  # from 608 + 64

    grid = ParameterGrid(param_grid)

    counter = 0
    mAP_max = 0
    best_params = ''
    for params in grid:
        print(params)

        get_detections_yolo4(images_source_dir=images_source_dir,
                             images_ext=images_ext,
                             txt_result_dir=txt_result_dir,
                             inference_image_size=params['inference_image_size'],
                             LABELS_FILE=LABELS_FILE,
                             CONFIG_FILE=CONFIG_FILE,
                             WEIGHTS_FILE=WEIGHTS_FILE,
                             MODEL_CONF_THR=params['MODEL_CONF_THR'],
                             NMS_THR=params['NMS_THR'])

        mAP = evaluate(groundtruths_dir=groundtruths_dir,
                       detections_dir=detections_dir,
                       image_size=image_size,
                       images_source_dir=images_source_dir,
                       images_ext=images_ext,
                       images_draw_path=images_draw_path,
                       draw_images=False)

        # mAP = 0.8763274628342
        filename = str(counter) + '_' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '_mAP_' + str(
            round(mAP, 4)) + '.txt'
        with open('logs/' + filename, 'w') as f:
            f.write(json.dumps(params))

        if mAP > mAP_max:
            mAP_max = mAP
            best_params = json.dumps(params)

        counter += 1

    # save best params
    filename = 'best_' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '_mAP_' + str(
        round(mAP_max, 4)) + '.txt'
    with open('logs/' + filename, 'w') as f:
        f.write(json.dumps(best_params))


if __name__ == '__main__':
    find_best_detection_params()
