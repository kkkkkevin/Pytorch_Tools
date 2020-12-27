import os
import json
import numpy as np

import cv2

from tqdm import tqdm


def convert_xyxy_to_cxcywh(boxes, w, h):
    _boxes = np.array(boxes, dtype=np.float)
    _boxes[:, 2] = _boxes[:, 2] - _boxes[:, 0]
    _boxes[:, 3] = _boxes[:, 3] - _boxes[:, 1]
    _boxes[:, 0] = _boxes[:, 0] + _boxes[:, 2] * 0.5
    _boxes[:, 1] = _boxes[:, 1] + _boxes[:, 3] * 0.5

    _boxes /= [w, h, w, h]
    _boxes = np.round(_boxes, 6)

    return _boxes


def conver_df_to_yolov5_format(df, classes, w, h):
    txt = ''
    for key, boxes in df['labels'].items():

        if key == 'Breezer School' or key == 'Jumper School':
            _boxes = convert_xyxy_to_cxcywh(boxes, w, h)

            for box in _boxes:
                txt += str(classes[key])
                txt += ' ' + str(box[0]) + ' ' + str(box[1]) + \
                    ' ' + str(box[2]) + ' ' + str(box[3])
                txt += '\n'
    return txt


def main(img_dir, ann_dir, save_path, classes):
    os.makedirs(save_path, exist_ok=True)

    imgs = list(sorted(os.listdir(img_dir)))
    anns = list(sorted(os.listdir(ann_dir)))

    for idx in tqdm(range(len(anns)), 'anno'):
        img_path = os.path.join(img_dir, imgs[idx])
        ann_path = os.path.join(ann_dir, anns[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape  # (h,w,c)

        with open(ann_path) as js:
            df = json.load(js)
            txt = conver_df_to_yolov5_format(df, classes, w, h)
            file_name = os.path.splitext(os.path.basename(anns[idx]))[0]
            with open(save_path + '/' + file_name + '.txt', mode='w') as tx:
                tx.write(txt)


if __name__ == '__main__':
    root = '/workspaces/data/ObjectDetection/school_of_fish/'

    img_dir = root + 'images/train_images'
    ann_dir = root + 'labels/train_annotations'

    save_dir = root + 'labels/train_annotations_yolov5'
    classes = {
        'Breezer School': 0,
        'Jumper School': 1,
        'Dolphin': 2,
        'Bird': 3,
        'Object': 4,
        'Cloud': 5,
        'Ripple': 6,
        'Smooth Surface': 7,
        'Wake': 8,
        'Each Fish': 9,
        'w': 10,
    }
    main(img_dir, ann_dir, save_dir, classes)
