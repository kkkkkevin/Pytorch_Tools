import os
from sklearn.model_selection import KFold


def get_fold_indices(imgs, anns):
    kf = KFold(5, shuffle=True)
    fold_list = []
    for train_index, test_index in kf.split(imgs):
        fold_list.append({
            'train_index': train_index,
            'test_index': test_index
        })

    return fold_list


def main(img_dir, ann_dir, save_path):

    imgs = list(sorted(os.listdir(img_dir)))
    anns = list(sorted(os.listdir(ann_dir)))

    fold_list = get_fold_indices(imgs, anns)

    fold_no = 0
    for fold in fold_list:
        train_fl = [img_dir + imgs[i] for i in fold['train_index']]
        test_fl = [img_dir + imgs[i] for i in fold['test_index']]
        with open(save_path + '/' + 'train' + f'{fold_no}' + '.txt', mode='w') as tx:
            tx.write('\n'.join(train_fl))
        with open(save_path + '/' + 'val' + f'{fold_no}' + '.txt', mode='w') as tx:
            tx.write('\n'.join(test_fl))
        fold_no += 1


if __name__ == '__main__':
    root = '/workspaces/data/ObjectDetection/school_of_fish/'

    img_dir = root + 'images/train_images/'
    ann_dir = root + 'labels/train_annotations/'
    save_path = root
    main(img_dir, ann_dir, save_path)
