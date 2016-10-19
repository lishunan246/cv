# coding=utf-8

import numpy as np
import cv2
import os
from LineSegment import LineSegment, LineSegmentGroup


# 条形码比例3:2

# 控制台输出单击处的颜色
# noinspection PyUnusedLocal
def on_mouse_click(event, x, y, flags, frame):
    if event == cv2.EVENT_LBUTTONDOWN:
        print frame[y, x]


yellow = (0, 255, 255)
thickness = 10

font = cv2.FONT_HERSHEY_SIMPLEX
input_path = 'input//'
output_path = 'output//'

for filename in os.listdir(input_path):

    img = cv2.imread(input_path + filename, cv2.IMREAD_COLOR)
    height, width, depth = img.shape

    gravy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gravy, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
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

    avg_theta = sum([ls.theta for ls in max_group]) / float(len(max_group))

    black = np.zeros((height, width, depth), np.uint8)

    for ls in max_group:  # 根据斜率找到的线段
        x1, y1, x2, y2 = ls.x1, ls.y1, ls.x2, ls.y2
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(black, (x1, y1), (x2, y2), (0, 255, 0), 2)
        ls.update(avg_theta)

    max_group.sort(key=lambda x: x.d)
    group_d = []

    for i in range(0, len(max_group)):
        ls = max_group[i]
        found_group = False
        for j in range(0, len(group_d)):
            if group_d[j].is_near(ls):
                group_d[j].insert(ls)
                found_group = True
                break
        if not found_group:
            new_g = LineSegmentGroup()
            new_g.insert(ls)
            group_d.append(new_g)

    cv2.putText(black, 'Group by d: ' + str(len(group_d)), (40, 200), font, 2, (255, 255, 255), 2)

    max_group = max(group_d, key=lambda x: len(x.lines)).lines

    x_max = max_group[0].x1
    x_min = max_group[len(max_group) - 1].x1

    y_min = min(map(lambda l: l.y1, max_group))
    y_max = max(map(lambda l: l.y2, max_group))

    cv2.line(img, (x_max, y_max), (x_max, y_min), yellow, thickness)
    cv2.line(img, (x_min, y_max), (x_min, y_min), yellow, thickness)
    cv2.line(img, (x_max, y_min), (x_min, y_min), yellow, thickness)
    cv2.line(img, (x_min, y_max), (x_max, y_max), yellow, thickness)

    cv2.line(black, (x_max, y_max), (x_max, y_min), yellow, thickness)
    cv2.line(black, (x_min, y_max), (x_min, y_min), yellow, thickness)
    cv2.line(black, (x_max, y_min), (x_min, y_min), yellow, thickness)
    cv2.line(black, (x_min, y_max), (x_max, y_max), yellow, thickness)

    cv2.imwrite(output_path + 'line_' + filename, img)
    cv2.imwrite(output_path + 'only_line_' + filename, black)
cv2.destroyAllWindows()
