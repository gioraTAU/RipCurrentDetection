import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def draw_rectangle_on_image(img, bbox):
    """Present image with bounding box on top

    Parameters
    ----------
    img : numpy.ndarray
        numpy image

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------

    open new figure with image and bounding box

    """
    pt1, pt2 = bbox[0, 0:2], bbox[0, 2:]
    pt3, pt4 = bbox[1, 0:2], bbox[1, 2:]
    cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=4)
    cv2.rectangle(img, pt1=pt3, pt2=pt4, color=(255, 0, 0), thickness=4)
    plt.imshow(img)
    plt.show()


def concat_imgs(img, label, angle, tar_path):

    h, w, c = img.shape
    target_size = (300, 300)
    label2 = np.zeros((1, 4))
    if angle == 0:
        im_cat = cv2.hconcat([img, img])
        label2[0, 0] = label[0] + w
        label2[0, 1] = label[1]
        label2[0, 2] = label[2] + w
        label2[0, 3] = label[3]
    else:
        im_cat = cv2.vconcat([img, img])
        label2[0, 0] = label[0]
        label2[0, 1] = label[1] + h
        label2[0, 2] = label[2]
        label2[0, 3] = label[3] + h

    img_resized = cv2.resize(im_cat, target_size)

    h_cat, w_cat, _ = im_cat.shape
    kx, ky = target_size[1] / w_cat, target_size[0] / h_cat

    box = np.concatenate((np.expand_dims(label, 0), label2), 0)

    col1 = np.expand_dims(np.round_(kx * box[:, 0]).astype(int), 1)
    col2 = np.expand_dims(np.round_(ky * box[:, 1]).astype(int), 1)
    col3 = np.expand_dims(np.round_(kx * box[:, 2]).astype(int), 1)
    col4 = np.expand_dims(np.round_(ky * box[:, 3]).astype(int), 1)

    bbox_sized = np.concatenate((col1, col2, col3, col4), axis=1)
    #draw_rectangle_on_image(img_resized, bbox_sized)
    return img_resized, bbox_sized


def create_doubleRips_dataset(labels_df, imgs_path, target_path):

    box = labels_df.loc[:, ['x1', 'y1', 'x2', 'y2']].to_numpy()
    imgs_name = labels_df['Name'].drop_duplicates()
    labels = labels_df['label']

    list_to_write = []
    for num, cur_img_fname in enumerate(imgs_name):

        if not(cur_img_fname.endswith(".png")):
            cur_img_fname = cur_img_fname + '.png'
            no_rip_flag = 1
        else:
            no_rip_flag = 0

        cur_img = imgs_path + cur_img_fname
        img = cv2.imread(cur_img)

        if not(img is None):

            if no_rip_flag == 0:
                cur_box = box[num, :]
                id = cur_img_fname.rfind('_')
                char = cur_img_fname[id + 1]
                f = char.isdigit()
                if f:
                    id2 = cur_img_fname.find('deg')
                    angle_str = cur_img_fname[id + 1:id2]
                    angle = int(angle_str)
                else:
                    angle = -90

                img_cat, bbox_sized = concat_imgs(img, cur_box, angle, target_path)

                cur_label = np.array((labels[num], labels[num]))

            else:
                img_cat = img
                bbox_sized = np.zeros((1, 4))[0].astype(int)
                cur_label = 0

            bbox_sized_to_write = bbox_sized
            list_to_write.append([cur_img_fname, bbox_sized[0][0], bbox_sized[0][1], bbox_sized[0][2], bbox_sized[0][3], labels[num]])

            cv2.imwrite(target_path + cur_img_fname, img_cat)

        else:

            print("Image " + cur_img_fname + " can't be found")

        if num % 100 == 0:
            print('Augment and resized picture number: ' + str(num))

    aug_data_labels = pd.DataFrame(list_to_write, columns=["Name", "x1", "y1", "x2", "y2", "label"])
    aug_data_labels.to_csv('doubleRip_aug_data_labels.csv', encoding='utf-8', index=False)


aug_imgs_path = '/home/giora/rip_current_detector/augmanted_training_data/'
target_path = '/home/giora/rip_current_detector/training_data_two_rips/'
labels_df = pd.read_csv('aug_data_labels.csv')

create_doubleRips_dataset(labels_df, aug_imgs_path, target_path)


box = labels_df.loc[:, ['x1', 'y1', 'x2', 'y2']].to_numpy()
imgs_name = labels_df['Name'].drop_duplicates()
labels = labels_df['label']
img_expl = aug_imgs_path + imgs_name[1]
img = cv2.imread(img_expl)
label = box[1, :]

id = img_expl.rfind('_')
char = img_expl[id+1]
f = char.isdigit()
if f:
    id2 = img_expl.find('deg')
    angle_str = img_expl[id+1:id2]
    angle = int(angle_str)
else:
    angle = -90

concat_imgs(img, label, angle, target_path)
