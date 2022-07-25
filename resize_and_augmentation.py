import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np


def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def rotate_image_pipe(image, alpha, bbox):
    """Rotate an image by alpha [deg] and rotate the bounding box as well

    Parameters
    ----------
    image : numpy.ndarray
        numpy image

    alpha : float
        angle by which the image is to be rotated

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------

    img_rot_resized: numpy.ndarray
        Rotated Image

    bboxes: numpy.ndarray
        Numpy array containing rotated bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """

    h, w = image.shape[0:2]
    bbox = np.expand_dims(bbox, 0)
    cx, cy = w // 2, h // 2
    img_rot = rotate_im(image, alpha)
    corners = get_corners(bbox)
    corners = np.hstack((corners, bbox[:, 4:]))

    corners[:, :8] = rotate_box(corners[:, :8], alpha, cx, cy, h, w)

    rotated_bbox = get_enclosing_box(corners)

    kx = w / img_rot.shape[1]

    ky = h / img_rot.shape[0]

    img_rot_resized = cv2.resize(img_rot, (w, h))

    rotated_bbox = np.round_((kx * rotated_bbox[:, 0], ky * rotated_bbox[:, 1], kx * rotated_bbox[:, 2], ky * rotated_bbox[:, 3])).astype(int).T

    bboxes = rotated_bbox

    bboxes = clip_box(bboxes, [0, 0, w, h], 0.25)
    if bboxes.shape[0] == 0:
        bboxes = np.zeros((1, 4))
    else:
        bboxes = bboxes[0]

    return img_rot_resized, bboxes


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
    pt1, pt2 = bbox[0:2], bbox[2:]
    cv2.rectangle(img, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=4)
    plt.imshow(img)
    plt.show()


def fix_size_and_aug(labels_file_path, target_size, imgs_path, target_path):
    """Fix all images exist on the specified csv file to be constant size and augment every image 3 times

    Parameters
    ----------
    labels_file_path: str
        A path contains the csv file path where all images name and bounding boxes are saved

    target_size: numpy.ndarray
        Numpy array  at size `1 X 2` where fixed size of the image is mentioned

    imgs_path : str
        A path contains the directory where all images are saved (both positive and negative labels)

    target_path : str
        A path contains the directory where all fixed size and augment images would be saved

    Returns
    -------

    Saves new file aug_data_labels.csv with names and label of the fixed and augment images and also
    save the fixed and augment images in te specified folder

    """

    labels_df = pd.read_csv(labels_file_path)
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

        #try:
        if not(img is None):
            cur_img_resized = cv2.resize(img, target_size)
            h, w, c = img.shape
            kx, ky = target_size[1] / w, target_size[0] / h

            original_box = box[num, :]
            bbox_sized = np.round_((kx * original_box[0], ky * original_box[1], kx * original_box[2], ky * original_box[3])).astype(int)

            cur_img_rot, bb_rot = rotate_image_pipe(cur_img_resized, 90, bbox_sized)
            cur_img_mrot, bb_mrot = rotate_image_pipe(cur_img_resized, -90, bbox_sized)

            fname_fixed0 = cur_img_fname[:-4] + '_0deg.png'
            fname_fixed90 = cur_img_fname[:-4] + '_90deg.png'
            fname_fixedm90 = cur_img_fname[:-4] + '_m90deg.png'
            cur_label = labels[num]

            if no_rip_flag == 1:
                bbox_sized, bb_rot, bb_mrot = np.zeros((1, 4))[0].astype(int), np.zeros((1, 4))[0].astype(int), np.zeros((1, 4))[0].astype(int)
                cur_label = 0
            list_to_write.append([fname_fixed0, bbox_sized[0], bbox_sized[1], bbox_sized[2], bbox_sized[3], cur_label])
            list_to_write.append([fname_fixed90, bb_rot[0], bb_rot[1], bb_rot[2], bb_rot[3], cur_label])
            list_to_write.append([fname_fixedm90, bb_mrot[0], bb_mrot[1], bb_mrot[2], bb_mrot[3], cur_label])

            cv2.imwrite(target_path + fname_fixed0, cur_img_resized)
            cv2.imwrite(target_path + fname_fixed90, cur_img_rot)
            cv2.imwrite(target_path + fname_fixedm90, cur_img_mrot)

        else:

            print("Image " + cur_img_fname + " can't be found")

        if num % 100 == 0:
            print('Augment and resized picture number: ' + str(num))

    aug_data_labels = pd.DataFrame(list_to_write, columns=["Name", "x1", "y1", "x2", "y2", "label"])
    aug_data_labels.to_csv('aug_data_labels.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    # S = np.array((300, 300))
    # data_labels_path = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\data_labels.csv'
    # im_path = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\training_data\\with_rips\\'
    # target_path = 'C:\\Giora\\TAU\\MSc_courses\\Deep_Learning\\final_project\\augmanted_training_data\\'
    # fix_size_and_aug(data_labels_path, S, im_path, target_path)
    from pathlib import Path

    data_path = Path('/home/giora/rip_current_detector')
    S = np.array((300, 300))
    data_labels_path = str(data_path / 'data_labels.csv')
    im_path = str(data_path / 'training_data') + '/'
    target_path = str(data_path / 'augmanted_training_data') + '/'
    fix_size_and_aug(data_labels_path, S, im_path, target_path)
