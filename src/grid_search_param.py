from sklearn.model_selection import ParameterGrid
from pathlib import Path
from evaluation import *
from datetime import datetime
import json


def find_best_detection_params():
    images_source_dir = Path('data/images/source')
    images_ext = ['*.jpg']
    txt_result_dir = Path('data/txt/detections')

    LABELS_FILE = 'model/yolo4/obj.names'
    CONFIG_FILE = 'model/yolo4/yolov4-obj-mycustom.cfg'
    WEIGHTS_FILE = 'model/yolo4/yolov4-obj-mycustom_best.weights'

    param_grid = {'MODEL_CONF_THR': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                  'NMS_THR': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                  'inference_image_size': [608, 672, 736, 800, 864, 928, 992]} # from 608 + 64

    grid = ParameterGrid(param_grid)

    counter = 0
    for params in grid:
        print(params)
        counter += 1
        # get_detections_yolo4(images_source_dir=images_source_dir,
        #                      images_ext=images_ext,
        #                      txt_result_dir=txt_result_dir,
        #                      inference_image_size=params['inference_image_size'],
        #                      LABELS_FILE=LABELS_FILE,
        #                      CONFIG_FILE=CONFIG_FILE,
        #                      WEIGHTS_FILE=WEIGHTS_FILE,
        #                      MODEL_CONF_THR=params['MODEL_CONF_THR'],
        #                      NMS_THR=params['NMS_THR'])
        #
        # groundtruths_dir = Path('data/txt/groundtruths')
        # detections_dir = Path('data/txt/detections')
        # images_source_dir = Path('data/images/source')
        # images_ext = ['*.jpg']
        # images_draw_path = Path('data/images/images_result_bboxes')
        # image_size = 1024
        # mAP = evaluate(groundtruths_dir=groundtruths_dir,
        #                detections_dir=detections_dir,
        #                image_size=image_size,
        #                images_source_dir=images_source_dir,
        #                images_ext=images_ext,
        #                images_draw_path=images_draw_path,
        #                draw_images=True)

        mAP = 0.8763274628342
        filename = str(counter) + '_' + datetime.today().strftime('%Y_%m_%d_%H_%M_%S') + '_' + str(
            round(mAP, 4)) + '.txt'
        with open('logs/' + filename, 'w') as f:
            f.write(json.dumps(params))

        data_log = []
        data_log.append('inference_image_size: ' + str(params['inference_image_size']))
        data_log.append('inference_image_size: ' + str(params['inference_image_size']))


if __name__ == '__main__':
    find_best_detection_params()
