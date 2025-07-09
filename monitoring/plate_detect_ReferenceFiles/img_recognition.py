# coding=gbk
import cv2
import numpy as np
from numpy.linalg import norm

SZ = 20  # ѵ��ͼƬ����
MAX_WIDTH = 1000  # ԭʼͼƬ�����
Min_Area = 2000  # ������������������
PROVINCE_START = 1000
# ����opencv��sample������svmѵ��
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11'] / m['mu02']
#     M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
#     img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
#     return img


# # ����opencv��sample������svmѵ��
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


provinces = [
    "zh_cuan", "��",
    "zh_e", "��",
    "zh_gan", "��",
    "zh_gan1", "��",
    "zh_gui", "��",
    "zh_gui1", "��",
    "zh_hei", "��",
    "zh_hu", "��",
    "zh_ji", "��",
    "zh_jin", "��",
    "zh_jing", "��",
    "zh_jl", "��",
    "zh_liao", "��",
    "zh_lu", "³",
    "zh_meng", "��",
    "zh_min", "��",
    "zh_ning", "��",
    "zh_qing", "��",
    "zh_qiong", "��",
    "zh_shan", "��",
    "zh_su", "��",
    "zh_sx", "��",
    "zh_wan", "��",
    "zh_xiang", "��",
    "zh_xin", "��",
    "zh_yu", "ԥ",
    "zh_yu1", "��",
    "zh_yue", "��",
    "zh_yun", "��",
    "zh_zang", "��",
    "zh_zhe", "��"
]

color_tr = {
    "green": ("����", "#55FF55"), 
    "yello": ("����", "#FFFF00"), 
    "blue": ("����", "#6666FF")
    }


