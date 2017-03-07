import numpy as np
# this is the main openCV class, the python binding file should be in /pythonXX/Lib/site-packages
import cv2
from matplotlib import pyplot as plt


def draw(image):
    # this is matplotlib solution (Figure 1)
    plt.imshow(image, 'gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


def load(filepath):
    return cv2.imread(filepath)


def canny(orig, thres1=200, thres2=150):
    canny_dst = orig.copy()
    # 輪郭を抽出する
    canny_dst = cv2.cvtColor(canny_dst, cv2.COLOR_BGR2GRAY)
    canny_dst = cv2.GaussianBlur(canny_dst, (5, 5), 0)
    return cv2.Canny(canny_dst, thres1, thres2)  # 50, 100


def contours(orig, mode=cv2.RETR_LIST, hierarchy=None):
    img = orig.copy()
    # 抽出した輪郭に近似する直線（？）を探す。
    img, cnts, hie = cv2.findContours(img, mode, cv2.CHAIN_APPROX_SIMPLE, hierarchy=hierarchy)
    # img2, cnts2, hie2 = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 面積が大きい順に並べ替える。
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts


def draw_contours(orig, orig_cnts, contour_ids=-1, thickness=cv2.FILLED):
    cnt_img = orig.copy()
    return cv2.drawContours(cnt_img, orig_cnts, contour_ids, (255, 255, 255), thickness=thickness)


def approx_poly(orig, orig_cnts):
    new_img = orig.copy()
    new_cnts = orig_cnts.copy()
    for k, cnt in enumerate(orig_cnts):
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        new_cnts[k] = cv2.approxPolyDP(cnt, epsilon, True)
    new_cnts.sort(key=cv2.contourArea, reverse=True)
    for cnt in new_cnts:
        for i in range(len(cnt)):
            cv2.line(new_img, tuple(cnt[i][0]), tuple(cnt[i - 1][0]), (255, 255, 255), 4)
    return new_img


def convex_poly(orig, orig_cnts):
    cc = orig.copy()
    # cv2.drawContours(cc, cnts, -1, (255,255,255), 2 );
    pts = []
    for cnt in orig_cnts:
        for pnt in cnt:
            pts.append(tuple(pnt[0]))
    convex_hull_points = cv2.convexHull(np.array(pts))

    cv2.fillConvexPoly(cc, convex_hull_points, (255, 255, 255))
    return cc


def concave_hull(orig_cnts):
    # // Prune contours
    max_area = 0.0
    for cnt in orig_cnts:
        if cv2.contourArea(cnt) >= max_area:
            max_area = cv2.contourArea(cnt)

    min_area = 0.20 * max_area
    pruned_contours = []
    for cnt in orig_cnts:
        if cv2.contourArea(cnt) >= min_area:
            pruned_contours.append(cnt)
    pruned_contours = np.array(pruned_contours)

    # // Smooth the contours
    smoothed_contours = pruned_contours.copy()
    gaussinan = cv2.transpose(cv2.getGaussianKernel(11, 4.0, cv2.CV_32FC1))
    for i, cnt in enumerate(smoothed_contours):
        x = []
        y = []
        for pnt in cnt:
            x.append(pnt[0][0])
            y.append(pnt[0][1])
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        x_smooth = cv2.filter2D(x, cv2.CV_32FC1, gaussinan)
        y_smooth = cv2.filter2D(y, cv2.CV_32FC1, gaussinan)
        for j in range(len(cnt)):
            cnt[j] = (x_smooth[j][0], y_smooth[j][0])
    return smoothed_contours


def erosion(orig, erosion_size, erosion_type=cv2.MORPH_RECT):
    e_dst = orig.copy()
    e_element = cv2.getStructuringElement(erosion_type, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                          (erosion_size, erosion_size))
    # /// Apply the erosion operation
    return cv2.erode(e_dst, e_element)


def dilation(orig, dilation_size, dilation_type=cv2.MORPH_RECT):
    d_dst = orig.copy()
    # cv2.MORPH_CROSS cv2.MORPH_ELLIPSE
    d_element = cv2.getStructuringElement(dilation_type,
                                          (2 * dilation_size + 1, 2 * dilation_size + 1),
                                          (dilation_size, dilation_size))
    # // / Apply the dilation operation
    return cv2.dilate(d_dst, d_element)
