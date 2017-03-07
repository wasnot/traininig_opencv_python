# -*- coding: utf-8 -*-
import numpy as np
# this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages
import cv2
from matplotlib import pyplot as plt


class Detector:
    def __init__(self, filepath, thres=15):
        # import image
        self.image = cv2.imread(filepath)
        self.gray = self.gray()
        self.closing = self.close_morphology(thres)
        self.without_contours, self.contours = self.remove_contours()
        self.only_contours = self.draw_contours()
        self.max_contour = self.max_contour()

    def gray(self):
        # change to grayscale
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def close_morphology(self, thres=15):
        # the value of 15 is chosen by trial-and-error to produce the best outline of the skull
        ret, thresh1 = cv2.threshold(self.gray, thres, 255, cv2.THRESH_BINARY)
        # square image kernel used for erosion
        kernel = np.ones((5, 5), np.uint8)
        # refines all edges in the binary image
        erosion = cv2.erode(thresh1, kernel, iterations=1)

        opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
        # this is for further removing small noises and holes in the image
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return closing

    def remove_contours(self):
        without_contours = self.closing.copy()
        # find contours with simple approximation
        without_contours, contours, hierarchy = cv2.findContours(without_contours, cv2.RETR_TREE,
                                                                 cv2.CHAIN_APPROX_SIMPLE)
        return without_contours, contours

    def draw_contours(self):
        image2 = self.without_contours.copy()
        image2 = cv2.drawContours(image2, self.contours, -1, (255, 255, 255), 4, cv2.FILLED)
        return image2

    def max_contour(self):
        img_area = self.image.shape[0] * self.image.shape[1]
        # list to hold all areas
        areas = []
        for contour in self.contours:
            # print('--')
            ar = cv2.contourArea(contour)
            if ar / img_area < 0.9:
                areas.append(ar)
                # print(str(ar))
        if not areas:
            return self.image
        max_area = max(areas)
        # print(max_area)
        # index of the list element with largest area
        max_area_index = areas.index(max_area)
        # largest area contour
        cnt = self.contours[max_area_index]

        image3 = self.image.copy()
        image3 = cv2.drawContours(image3, [cnt], 0, (255, 255, 255), 3, maxLevel=0)
        return image3

    @staticmethod
    def detect(filepath, thres=15):
        detector = Detector(filepath, thres)

        Detector.draw(detector.gray)
        Detector.draw(detector.closing)
        Detector.draw(detector.only_contours)
        Detector.draw(detector.max_contour)

    @staticmethod
    def draw(image):
        # this is matplotlib solution (Figure 1)
        plt.imshow(image, 'gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

        # cv2.imshow('gwash', gwashBW) #this is for native openCV display
        # cv2.waitKey(0)
