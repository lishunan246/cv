# coding=utf-8

import numpy as np
import cv2
import math
import os
from LineSegment import LineSegment, LineSegmentGroup


# 条形码比例3:2

yellow = (0, 255, 255)
thickness = 10
font = cv2.FONT_HERSHEY_SIMPLEX
input_path = 'input//'
output_path = 'output//'

for filename in os.listdir(input_path):
    original = cv2.imread(input_path + filename, cv2.IMREAD_COLOR)
    img = original.copy()
    height, width, depth = img.shape

    gravy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gravy, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    line_group = []
    black = np.zeros((height, width, depth), np.uint8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ls = LineSegment(x1, y1, x2, y2)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(black, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(line_group) == 0:  # first line group
            line_group.append([ls])
        else:
            done = False
            for i in range(0, len(line_group)):
                if abs(line_group[i][0].theta - ls.theta) < 0.1:  # 比较斜率
                    line_group[i].append(ls)
                    done = True
                    break

            if not done:
                line_group.append([ls])

    max_group = max(line_group, key=lambda x: len(x))

    avg_theta = sum([ls.theta for ls in max_group]) / float(len(max_group))

    for ls in max_group:
        x1, x2, y1, y2 = ls.x1, ls.x2, ls.y1, ls.y2
        # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # cv2.line(black, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # 根据斜率找到的线段
        ls.update(avg_theta)

    max_group.sort(key=lambda x: x.d)
    group_d = []

    black2 = np.zeros((height, width, depth), np.uint8)
    for i in range(0, len(max_group)):
        ls = max_group[i]
        x1, x2, y1, y2 = ls.x1, ls.x2, ls.y1, ls.y2
        cv2.line(black2, (x1, y1), (x2, y2), (0, 0, 255), 3)
        #cv2.waitKey()
        #cv2.imshow("lines", cv2.resize(black2,(0,0),fx=0.2,fy=0.2))

        j = len(group_d) - 1
        if j < 0:
            new_g = LineSegmentGroup()
            new_g.insert(ls)
            group_d.append(new_g)
        else:
            if group_d[j].is_near(ls):
                group_d[j].insert(ls)
            else:
                new_g = LineSegmentGroup()
                new_g.insert(ls)
                group_d.append(new_g)

    cv2.putText(black, 'Group by d: ' + str(len(group_d)), (40, 200), font, 2, (255, 255, 255), 2)

    max_group = max(group_d, key=lambda x: len(x.lines)).lines
    for ls in max_group:
        x1, x2, y1, y2 = ls.x1, ls.x2, ls.y1, ls.y2
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.line(black, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # 根据斜率 & x找到的线段
        # ls.update(avg_theta)

    x_max = max_group[0].x1
    x_min = max_group[len(max_group) - 1].x1

    y_min = min(map(lambda l: l.y1, max_group))
    y_max = max(map(lambda l: l.y2, max_group))

    x_avg = (x_max + x_min) / 2
    y_avg = (y_max + y_min) / 2

    pts1 = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(original, M, (300, 200))

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
    cv2.imwrite(output_path + 'dst_' + filename, dst)
cv2.destroyAllWindows()
