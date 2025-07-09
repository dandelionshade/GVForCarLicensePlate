# coding=gbk
import os
import cv2
from PIL import Image
import numpy as np

import img_math 
import img_recognition 

SZ = 20  # ѵ��ͼƬ����
MAX_WIDTH = 1000  # ԭʼͼƬ�����
Min_Area = 2000  # ������������������
PROVINCE_START = 1000


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # # ѵ��svm
    # def train(self, samples, responses):
    #     self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # �ַ�ʶ��
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:
    def __init__(self):
        pass

    def train_svm(self):
        # ʶ��Ӣ����ĸ������
        self.model = SVM(C=1, gamma=0.5)
        # ʶ������
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")

    def img_first_pre(self, car_pic_file):
        """
        :param car_pic_file: ͼ���ļ�
        :return:�Ѿ�����õ�ͼ���ļ� ԭͼ���ļ�
        """
        if type(car_pic_file) == type(""):
            img = img_math.img_read(car_pic_file)   #��ȡ�ļ�
        else:
            img = car_pic_file

        pic_hight, pic_width = img.shape[:2]  #ȡ��ɫͼƬ�ĸߡ���
        if pic_width > MAX_WIDTH:
            resize_rate = MAX_WIDTH / pic_width
            # ��СͼƬ
            img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        # ����interpolation �м�����������ѡ��:
        # cv2.INTER_AREA - �ֲ������ز������ʺ���СͼƬ��
        # cv2.INTER_CUBIC�� cv2.INTER_LINEAR ���ʺϷŴ�ͼ������INTER_LINEARΪĬ�Ϸ�����
        # ��˹�˲���һ������ƽ���˲������ڳ�ȥ��˹�����кܺõ�Ч��

        temp = img
        # img = cv2.GaussianBlur(img, (5, 5), 0)
        oldimg = img
        # ת���ɻҶ�ͼ��
        # ת����ɫ�ռ� cv2.cvtColor
        # BGR ---> Gray  cv2.COLOR_BGR2GRAY
        # BGR ---> HSV  cv2.COLOR_BGR2HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        cv2.imwrite("source/tmp/img_gray.jpg", img)
        
        #ones()����һ��ȫ1��nά���� 
        Matrix = np.ones((20, 20), np.uint8)  

        # ������:�Ƚ��Ը�ʴ�ٽ������;ͽ��������㡣��������ȥ�������� cv2.MORPH_OPEN        
        img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, Matrix)

        # ͼƬ�������ں�
        # g (x) = (1 ? ��)f0 (x) + ��f1 (x)   a����0��1����ͬ��aֵ����ʵ�ֲ�ͬ��Ч��
        img_opening = cv2.addWeighted(img, 1, img_opening, -1, 0)
        # cv2.imwrite("tmp/img_opening.jpg", img_opening)
        # ����20*20��Ԫ��Ϊ1�ľ��� ������������img�غ�


        # Otsu��s��ֵ��
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Canny ��Ե���
        # �ϴ����ֵ2���ڼ��ͼ�������Եı�Ե  һ������¼���Ч��������ô��������Ե�������Ƕ϶�������
        # ��С����ֵ1���ڽ���Щ��ϵı�Ե��������
        img_edge = cv2.Canny(img_thresh, 100, 200)
        cv2.imwrite("source/tmp/img_edge.jpg", img_edge)

        Matrix = np.ones((4, 19), np.uint8)
        # ������:�������ٸ�ʴ
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, Matrix)
        # ������
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix)
        cv2.imwrite("source/tmp/img_xingtai.jpg", img_edge2)
        return img_edge2, oldimg
    
    def img_only_color(self, filename, oldimg, img_contours):
        """
        :param filename: ͼ���ļ�
        :param oldimg: ԭͼ���ļ�
        :return: ʶ�𵽵��ַ�����λ�ĳ���ͼ�񡢳�����ɫ
        """
        pic_hight, pic_width = img_contours.shape[:2] # #ȡ��ɫͼƬ�ĸߡ���

        lower_blue = np.array([100, 110, 110])
        upper_blue = np.array([130, 255, 255])
        lower_yellow = np.array([15, 55, 55])
        upper_yellow = np.array([50, 255, 255])
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 255, 255])

        # BGR ---> HSV
        hsv = cv2.cvtColor(filename, cv2.COLOR_BGR2HSV)
        # ����cv2.inRange��������ֵ��ȥ����������
        # ����1��ԭͼ
        # ����2��ͼ���е���ֵ��ͼ��ֵ��Ϊ0
        # ����3��ͼ���и���ֵ��ͼ��ֵ��Ϊ0
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv, lower_yellow, upper_green)

        # ͼ����������  ��λ���� ��λ�����У� AND�� OR�� NOT�� XOR ��
        output = cv2.bitwise_and(hsv, hsv, mask=mask_blue + mask_yellow + mask_green)
        # ������ֵ�ҵ���Ӧ��ɫ

        output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        Matrix = np.ones((20, 20), np.uint8)
        #ʹ��һ�� 20x20 �ľ����
        img_edge1 = cv2.morphologyEx(output, cv2.MORPH_CLOSE, Matrix)  #������
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, Matrix) #������

        card_contours = img_math.img_findContours(img_edge2)
        card_imgs = img_math.img_Transform(card_contours, oldimg, pic_width, pic_hight)
        colors, car_imgs = img_math.img_color(card_imgs)
        print(colors)

        predict_result = []
        predict_str = ""
        roi = None
        card_color = None

        for i, color in enumerate(colors):
            try:
                if color in ("blue", "yello", "green"):
                    card_img = card_imgs[i]

                    try:
                        gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite("source/tmp/chepai_GRAY.jpg", gray_img)
                    except:
                        print("grayת��ʧ��")

                    # �ơ��̳����ַ��ȱ��������������Ƹպ��෴�����Իơ��̳�����Ҫ����
                    if color == "green" or color == "yello":
                        gray_img = cv2.bitwise_not(gray_img)
                    ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    x_histogram = np.sum(gray_img, axis=1)

                    x_min = np.min(x_histogram)
                    x_average = np.sum(x_histogram) / x_histogram.shape[0]
                    x_threshold = (x_min + x_average) / 2
                    wave_peaks = img_math.find_waves(x_threshold, x_histogram)
                    
                    if len(wave_peaks) == 0:
                        # print("peak less 0:")
                        continue
                    # ��Ϊˮƽ�������Ĳ���Ϊ��������
                    wave = max(wave_peaks, key=lambda x: x[1] - x[0])

                    
                    gray_img = gray_img[wave[0]:wave[1]]
                    # ���Ҵ�ֱֱ��ͼ����
                    row_num, col_num = gray_img.shape[:2]
                    # ȥ���������±�Ե1�����أ�����ױ�Ӱ����ֵ�ж�
                    gray_img = gray_img[1:row_num - 1]
                    y_histogram = np.sum(gray_img, axis=0)
                    y_min = np.min(y_histogram)
                    y_average = np.sum(y_histogram) / y_histogram.shape[0]
                    y_threshold = (y_min + y_average) / 5  # U��0Ҫ����ֵƫС������U��0�ᱻ�ֳ�����
                    wave_peaks = img_math.find_waves(y_threshold, y_histogram)
                    if len(wave_peaks) < 6:
                        # print("peak less 1:", len(wave_peaks))
                        continue

                    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
                    max_wave_dis = wave[1] - wave[0]
                    # �ж��Ƿ�����೵�Ʊ�Ե
                    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                        wave_peaks.pop(0)

                    # ��Ϸ��뺺��
                    cur_dis = 0
                    for i, wave in enumerate(wave_peaks):
                        if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                            break
                        else:
                            cur_dis += wave[1] - wave[0]
                    if i > 0:
                        wave = (wave_peaks[0][0], wave_peaks[i][1])
                        wave_peaks = wave_peaks[i + 1:]
                        wave_peaks.insert(0, wave)

                    point = wave_peaks[2]
                    point_img = gray_img[:, point[0]:point[1]]
                    if np.mean(point_img) < 255 / 5:
                        wave_peaks.pop(2)

                    if len(wave_peaks) <= 6:
                        # print("peak less 2:", len(wave_peaks))
                        continue
                    # print(wave_peaks)
                    
                    # wave_peaks  �����ַ� �����б� ����7������ʼ�ĺ����꣬�����ĺ����꣩



                    part_cards = img_math.seperate_card(gray_img, wave_peaks)

                    for i, part_card in enumerate(part_cards):
                        # �����ǹ̶����Ƶ�í��

                        if np.mean(part_card) < 255 / 5:
                            # print("a point")
                            continue
                        part_card_old = part_card

                        w = abs(part_card.shape[1] - SZ) // 2

                        part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                        
                        part_card = img_recognition.preprocess_hog([part_card])
                        if i == 0:
                            resp = self.modelchinese.predict(part_card)
                            charactor = img_recognition.provinces[int(resp[0]) - PROVINCE_START]
                        else:
                            resp = self.model.predict(part_card)
                            charactor = chr(resp[0])
                        # �ж����һ�����Ƿ��ǳ��Ʊ�Ե�����賵�Ʊ�Ե����Ϊ��1
                        if charactor == "1" and i == len(part_cards) - 1:
                            if part_card_old.shape[0] / part_card_old.shape[1] >= 7:  # 1̫ϸ����Ϊ�Ǳ�Ե
                                continue
                        predict_result.append(charactor)
                        predict_str = "".join(predict_result)

                    roi = card_img
                    card_color = color
                    break
            except:
                pass
        cv2.imwrite("source/tmp/img_caijian.jpg", roi)
        return predict_str, roi, card_color  # ʶ�𵽵��ַ�����λ�ĳ���ͼ�񡢳�����ɫ
