import os
import json
import numpy as np
import pandas as pd
import cv2

from tqdm.autonotebook import tqdm


def convert_xyxy_to_cxcywh(boxes, w, h):
    x = np.array(boxes, dtype=np.float32)
    _boxes = x.copy()
    _boxes[:, 0] = (x[:, 0] + x[:, 2]) * 0.5
    _boxes[:, 1] = (x[:, 1] + x[:, 3]) * 0.5
    _boxes[:, 2] = x[:, 2] - x[:, 0]
    _boxes[:, 3] = x[:, 3] - x[:, 1]

    _boxes /= [w, h, w, h]
    _boxes = np.round(_boxes, 6)

    return _boxes


def conver_json2txt(df, classes, w, h,
                    target_keys, trans_box,
                    seq=' '):
    txt = ''
    for key, boxes in df['labels'].items():

        if key in target_keys:
            if trans_box == 'xyxy2yolo':
                _boxes = convert_xyxy_to_cxcywh(boxes, w, h)
            else:
                _boxes = boxes
            # _boxes = boxes
            for box in _boxes:
                txt += str(classes[key])
                txt += seq + str(box[0]) + seq + str(box[1]) + \
                    seq + str(box[2]) + seq + str(box[3])
                txt += '\n'
    return txt


def conver_json2csv(js, img_id, ann_id,
                    classes, w, h,
                    target_keys, trans_box):
    df_list = []
    for key, boxes in js['labels'].items():
        if key in target_keys:
            if trans_box == 'xyxy2yolo':
                _boxes = convert_xyxy_to_cxcywh(boxes, w, h)
            else:
                _boxes = boxes

            for box in _boxes:
                df_list.append([img_id, classes[key], w, h, box, ann_id])

    return df_list


def create_csv(img_dir, ann_dir, save_path, classes,
               target_keys, trans_box):

    df_list = []
    os.makedirs(save_path, exist_ok=True)

    imgs = list(sorted(os.listdir(img_dir)))
    anns = list(sorted(os.listdir(ann_dir)))

    for idx in tqdm(range(len(imgs)), 'anno'):
        img_path = os.path.join(img_dir, imgs[idx])
        ann_path = os.path.join(ann_dir, anns[idx])

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        h, w, _ = img.shape  # (h,w,c)

        with open(ann_path) as js:
            js = json.load(js)
            tmp = conver_json2csv(js, imgs[idx], anns[idx],
                                  classes, w, h, target_keys, trans_box)
            df_list.extend(tmp)

    col = ['image_id', 'label_id', 'width', 'height', 'bbox', 'ann_id']
    df = pd.DataFrame(df_list, columns=col)

    file_name = 'train'
    df.to_csv(f'{save_path}/{file_name}.csv')


def create_text(img_dir, ann_dir, save_path, classes,
                target_keys, trans_box):
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
            txt = conver_json2txt(df, classes, w, h, target_keys, trans_box)
            file_name = os.path.splitext(os.path.basename(anns[idx]))[0]
            with open(save_path + '/' + file_name + '.txt', mode='w') as tx:
                tx.write(txt)


if __name__ == '__main__':
    root = '/workspaces/data/ObjectDetection/school_of_fish/'

    img_dir = root + 'images/train_images'
    ann_dir = root + 'labels/train_annotations'

    save_dir = root + 'labels'
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
    # target class
    target_keys = ['Breezer School', 'Jumper School']
    # xyxy to coco, yolo
    trans_box = ''  # keys 'xyxy2coco', 'xyxy2yolo'
    create_csv(img_dir, ann_dir, save_dir, classes, target_keys, trans_box)
