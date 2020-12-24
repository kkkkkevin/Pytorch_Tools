import IPython
import os
import json
import random
import numpy as np
import requests
import collections as cl
import cv2
from io import BytesIO
from math import trunc
from PIL import Image as PILImage
from PIL import ImageDraw as PILImageDraw
import base64
from tqdm import tqdm


mask_id_rgb = [
    # [R, G, B]
    # able to drive
    [128, 64, 128],  # 0: road
    [255, 128, 128],  # 1: dirt road
    [190, 153, 153],  # 2: parking lot
    [102, 102, 156],  # 3: drive way
    [152, 251, 152],  # 4: other road marking
    [255, 0, 0],  # 5: rail
    [0, 60, 100],  # 6: lane boundary
    [0, 80, 100],  # 7: other lane marking
    [120, 240, 120],  # 8: puddle
    [128, 0, 255],  # 9: rut
    # unable to drive
    [0, 0, 70],  # 10: other obstacle
    [70, 130, 180],  # 11: sky
    [220, 20, 60],  # 12: person
    [119, 11, 32],  # 13: two-wheel vehicle
    [0, 0, 142],  # 14: car
    [220, 220, 0],  # 15: traffic sign
    [70, 70, 70],  # 16: building
    [90, 120, 130],  # 17: crack
    [255, 255, 255],  # 18: snow
]


def info(description="Test",
         url="https://test",
         version="0.01",
         year=2019,
         contributor="xxxx",
         data_created="2019/09/10"):

    tmp = cl.OrderedDict()
    tmp["description"] = description
    tmp["url"] = url
    tmp["version"] = "version"
    tmp["year"] = year
    tmp["contributor"] = contributor
    tmp["data_created"] = data_created

    return tmp


def licenses(id=1,
             url='dummy_words',
             name="administrater"):

    tmp = cl.OrderedDict()
    tmp["id"] = id
    tmp["url"] = url
    tmp["name"] = name

    return tmp


def images(license,
           img_path,
           img_list,):

    tmps = []
    for i in tqdm(range(len(img_list)), 'images'):
        img = cv2.imread(img_path + img_list[i], cv2.IMREAD_COLOR)
        h, w, _ = img.shape  # (h,w,c)

        tmp = cl.OrderedDict()
        tmp["license"] = 0
        tmp["id"] = i
        tmp["file_name"] = img_list[i]
        tmp["width"] = w
        tmp["height"] = h
        tmp["date_captured"] = "none"
        tmp["coco_url"] = 'none'
        tmp["flickr_url"] = 'none'
        tmps.append(tmp)

    return tmps


def segmantation(mask):
    segms = []
    boxes = []
    category_ids = []

    for i in tqdm(range(len(mask_id_rgb)), 'RGB_ID'):
        pos = np.where((mask == mask_id_rgb[i]).all(axis=2))

        if len(pos[0]) > 0:
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))

            segms.append(np.array(pos).transpose().tolist())
            boxes.append([xmin, ymin, xmax, ymax])
            category_ids.append(i)

    return segms, boxes, category_ids


def annotations(mask_path,
                mask_list):
    tmps = []

    for img_idx in tqdm(range(len(mask_list)), 'Segm'):  # len(mask_list)
        mask = cv2.imread(mask_path + mask_list[img_idx], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        segms, boxes, category_ids = segmantation(mask)

        assert len(segms) == len(boxes), f'wrong length'

        for segm_idx in range(len(segms)):
            tmp = cl.OrderedDict()
            tmp["segmentation"] = segms[segm_idx]
            tmp["id"] = int(str(1000) + str(segm_idx))
            tmp["image_id"] = img_idx
            tmp["category_id"] = category_ids[segm_idx]
            tmp["area"] = 10
            tmp["iscrowd"] = 0
            tmp["bbox"] = boxes[segm_idx]
            tmps.append(tmp)

    return tmps


def categories():
    tmps = []
    sup = ["animal", "pill"]
    cat = ["dog", "allergy"]
    for i in range(2):
        tmp = cl.OrderedDict()
        tmp["id"] = str(i)
        tmp["supercategory"] = sup[i]
        tmp["name"] = cat[i]
        tmps.append(tmp)
    return tmps


def main(query_list, img_path, mask_path):
    img_list = list(sorted(os.listdir(img_path)))
    mask_list = list(sorted(os.listdir(mask_path)))

    assert len(img_list) == len(
        mask_list), f'元画像とマスク画像の枚数が違う.img:{len(img_list)},mask:{len(mask_list)}'
    js = cl.OrderedDict()
    for i in tqdm(range(len(query_list)), 'Query'):
        tmp = ""
        # Info
        if query_list[i] == "info":
            tmp = info()
        # Licenses
        if query_list[i] == "licenses":
            tmp = licenses()
        # Images
        if query_list[i] == "images":
            tmp = images(0, img_path, img_list)
        # Annotations
        if query_list[i] == "annotations":
            tmp = annotations(mask_path, mask_list)
        # Categories
        if query_list[i] == "categories":
            tmp = categories()
        # save it
        js[query_list[i]] = tmp

    # write
    fw = open('datasets.json', 'w')
    json.dump(
        js,
        fw,
        ensure_ascii=False,
        indent=4,
        sort_keys=True,
        separators=(
            ',',
            ': '))


if __name__ == '__main__':
    root = '/workspaces/data/Segmentation/' + 'Offload/'
    image_path = root + 'train_images_A/'
    mask_path = root + 'train_annotations_A/'
    query_list = ["info", "licenses", "images", "annotations", "categories"]
    main(query_list, image_path, mask_path)
