# Imports
import cv2
import numpy as np

# Global Variables
CITYSCAPES_SS = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (244, 35, 232),
    3: (70, 70, 70),
    4: (102, 102, 156),
    5: (190, 153, 153),
    6: (153, 153, 153),
    7: (250, 170, 30),
    8: (220, 220, 0),
    9: (107, 142, 35),
    10: (152, 251, 152),
    11: (70, 130, 180),
    12: (220, 20, 60),
    13: (255, 0, 0),
    14: (0, 0, 142),
    15: (0, 0, 70),
    16: (0, 60, 100),
    17: (0, 80, 100),
    18: (0, 0, 230),
    19: (119, 11, 32),
    21: (110, 190, 160),
    22: (170, 120, 50),
    23: (55, 90, 80),
    24: (45, 60, 150),
    25: (157, 234, 50),
    26: (81, 0, 81),
    27: (150, 100, 100),
    28: (230, 150, 140),
    29: (180, 165, 180),
}

CITYSCAPES_ID = {CITYSCAPES_SS[key]: key for key in CITYSCAPES_SS.keys()}


# Functions
def rgb2ss(rgb_img):
    rgb_img = np.asarray(rgb_img, dtype=np.int64)
    w, h, ch = np.shape(rgb_img)
    n_classes = len(CITYSCAPES_SS.keys())

    ss_img = np.asarray([CITYSCAPES_ID[tuple(rgb_img[u, v, :])] for u in range(w) for v in range(h)])
    ss_img_oh = np.zeros((w * h, n_classes + 1))
    ss_img_oh[range(w * h), ss_img] = 1
    ss_img_oh = np.reshape(ss_img_oh, (w, h, n_classes + 1))

    return ss_img_oh


def ss2rgb(ss_img):
    w, h, _ = np.shape(ss_img)
    id_img = np.argmax(ss_img, axis=2)

    rgb_img = np.asarray([CITYSCAPES_SS[id_img[u, v]] for u in range(w) for v in range(h)])
    rgb_img = np.reshape(rgb_img, (w, h, 3))
    rgb_img = np.asarray(rgb_img, dtype=np.uint8)

    return rgb_img

# Testing main
if __name__ == '__main__':
    img_name = 'D:/DeepLearning/datasets/rosbag2_2025_05_08-12_07_12/ss_bev/ss_bev_1746698832_941827297.png'

    rgb_img = cv2.imread(img_name)[..., ::-1]
    ss_img = rgb2ss(rgb_img)
    rgb_img_rec = ss2rgb(ss_img)

    id_img = np.argmax(ss_img, axis=2)
    id_img = np.asarray(id_img * 10., dtype=np.uint8)

    cv2.imshow('semantic ids', id_img)
    cv2.imshow('rgb img reconstructed', rgb_img_rec)
    cv2.waitKey()
