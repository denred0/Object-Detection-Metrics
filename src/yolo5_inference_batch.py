from os import listdir
from os.path import isfile, join
from time import time
import json
import torch
from PIL import Image
import sys


def run(input_path, output_path):
    images_names = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    weights_path = 'model_folder/yolo5/yolov5l6_f1_512_half.pt'

    model = torch.hub.load('model_folder/yolo5/yolov5_master', 'custom', path=weights_path, source='local')
    model.half()

    results_output = []

    time_start = time()
    batch_size = 16
    for i in range(0, len(images_names), batch_size):
        if i + batch_size >= len(images_names):
            end_ind = len(images_names)
        else:
            end_ind = i + batch_size

        batch = []
        for j in range(i, end_ind, 1):
            img = join(input_path, images_names[j])
            batch.append(img)

        results = model(batch, augment=True)
        results_np = results.xyxy
        for k, result_np in enumerate(results_np):
            result_np = result_np.detach().cpu().numpy()
            for box in result_np:
                box_result = {}
                box_result['image_id'] = int(results.files[k].replace('.jpg', ''))
                box_result['category_id'] = int(box[5])
                box_result['score'] = float(box[4])
                box_result['bbox'] = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
                results_output.append(box_result)

        # results.print()

    with open(output_path, "w") as fp:
        fp.write(json.dumps(results_output))

    print('\n' + json.dumps(results_output))


if __name__ == '__main__':
    # input_path = sys.argv[1]
    # output_path = sys.argv[2]
    output_path = 'predict_yolo.json'
    input_path = 'data/yolo5_inference/images_batch'

    run(input_path, output_path)
