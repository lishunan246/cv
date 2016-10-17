# coding=utf-8
import numpy as np
import cv2
from LineSegment import LineSegment


# 控制台输出单击处的颜色
# noinspection PyUnusedLocal
def on_mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        print frame[y, x]


lower_blue = np.array([100, 20, 20])
upper_blue = np.array([125, 255, 255])
kernel = np.ones((2, 2), np.uint8)

img = cv2.imread('input//6.jpg', cv2.IMREAD_COLOR)
cv2.imshow('image', img)

height, width, depth = img.shape
cv2.setMouseCallback('Image', on_mouse_click, img)

while True:
    key = 0xff & cv2.waitKey()
    if key == 27:
        break
    elif key == ord('r'):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = img.copy()
        mask = cv2.inRange(img, lower_blue, upper_blue)
        mask = cv2.bitwise_not(mask)
        cv2.imshow("mask", mask)

        res = cv2.bitwise_and(img, img, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        # cv2.imshow('paint', res)
        res = cv2.inpaint(res, cv2.dilate(cv2.bitwise_not(mask), kernel, iterations=1), 3, cv2.INPAINT_TELEA)
        # cv2.imshow('inpaint', res)
    elif key == ord('g'):
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
        line_group = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ls = LineSegment(x1, y1, x2, y2)
            if len(line_group) == 0:  # first line group
                line_group.append([ls])
            else:
                done = False
                for i in range(0, len(line_group)):
                    if abs(np.arctan(line_group[i][0].k) - np.arctan(ls.k)) < 0.05:  # 比较斜率
                        line_group[i].append(ls)
                        done = True
                        break

                if not done:
                    line_group.append([ls])

        max_group = max(line_group, key=lambda x: len(x))
        for ls in max_group:
            x1, y1, x2, y2 = ls.x1, ls.y1, ls.x2, ls.y2
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow('houghlines5.jpg', img)
    elif key == ord('f'):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([]), np.array([]))

cv2.destroyAllWindows()
