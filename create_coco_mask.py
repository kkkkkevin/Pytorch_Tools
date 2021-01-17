import os
import numpy as np
import cv2
from tqdm.autonotebook import tqdm

mask_id_rgb_categy_all = [
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

mask_id_rgb_categy_target = [
    # [R, G, B]
    # able to drive
    [128, 64, 128],  # 0: road
    [255, 128, 128],  # 1: dirt road
    [0, 0, 70],  # 10: other obstacle
]


def remask(mask, mask_id_rgb):
    assert (len(mask.shape) > 2 and mask.dtype ==
            'uint8'), 'please input rgb img with uint8 dtype'

    assert len(
        mask_id_rgb) < 255, 'The maximum number of IDs has been exceeded.(ids<255)'

    h, w, _ = mask.shape
    remask = np.full((h, w), 255, dtype=np.uint8)

    for i in range(len(mask_id_rgb)):
        pos = np.where((mask == mask_id_rgb[i]).all(axis=2))
        remask[pos] = i

    # assert len(np.where(remask == 255)[
    #           0]) == 0, 'There are RGB pixels with undefined ID.'

    return remask


def main(mask_path, save_path, mask_id_rgb):

    mask_list = list(sorted(os.listdir(mask_path)))

    for img_idx in tqdm(range(len(mask_list)), 'Segm'):  # len(mask_list)
        mask = cv2.imread(mask_path + mask_list[img_idx], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB)
        _remask = remask(mask, mask_id_rgb)

        save_name = save_path + \
            os.path.splitext(os.path.basename(mask_list[img_idx]))[0] + '_remask.png'
        cv2.imwrite(save_name, _remask)


if __name__ == '__main__':
    root = '/workspaces/data/Segmentation/' + 'Offload/'

    mask_path = root + 'train_annotations_A/'
    save_path = '/workspaces/data/Segmentation/Offload/train_annotations_A_remask_3category/'

    main(mask_path, save_path, mask_id_rgb_categy_target)
