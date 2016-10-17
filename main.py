# coding=utf-8

import numpy as np
import cv2
import os
import math
from LineSegment import LineSegment


# 条形码比例3:2

# 控制台输出单击处的颜色
# noinspection PyUnusedLocal
def on_mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        print frame[y, x]


lower_blue = np.array([100, 20, 20])
upper_blue = np.array([125, 255, 255])
kernel = np.ones((2, 2), np.uint8)

input_path = 'input//'
output_path = 'output//'

for filename in os.listdir(input_path):

    img = cv2.imread(input_path + filename, cv2.IMREAD_COLOR)
    height, width, depth = img.shape

    gravy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gravy, 50, 150, apertureSize=3)
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
                if abs(line_group[i][0].theta - ls.theta) < 0.05:  # 比较斜率
                    line_group[i].append(ls)
                    done = True
                    break

            if not done:
                line_group.append([ls])

    max_group = max(line_group, key=lambda x: len(x))

    max_group.sort(key=lambda x: x.x1)

    black = np.zeros((height, width, depth), np.uint8)

    for ls in max_group:
        x1, y1, x2, y2 = ls.x1, ls.y1, ls.x2, ls.y2
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(black, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x_max = max_group[0].x1
    x_min = max_group[len(max_group) - 1].x1

    y1s = map(lambda l: l.y1, max_group)
    y_min = min(y1s)

    y_max = max(map(lambda l: l.y2, max_group))

    cv2.line(img, (x_max, y_max), (x_max, y_min), (255, 0, 0), 2)
    cv2.line(img, (x_min, y_max), (x_min, y_min), (255, 0, 0), 2)
    cv2.line(img, (x_max, y_min), (x_min, y_min), (255, 0, 0), 2)
    cv2.line(img, (x_min, y_max), (x_max, y_max), (255, 0, 0), 2)

    cv2.line(black, (x_max, y_max), (x_max, y_min), (0, 0, 255), 2)
    cv2.line(black, (x_min, y_max), (x_min, y_min), (0, 0, 255), 2)
    cv2.line(black, (x_max, y_min), (x_min, y_min), (0, 0, 255), 2)
    cv2.line(black, (x_min, y_max), (x_max, y_max), (0, 0, 255), 2)

    cv2.imwrite(output_path + 'line_' + filename, img)
    cv2.imwrite(output_path + 'only_line_' + filename, black)
cv2.destroyAllWindows()
