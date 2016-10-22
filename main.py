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
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.line(black, (x1, y1), (x2, y2), (0, 255, 0), 1)
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
        # 根据斜率找到的线段
        ls.update(avg_theta)

    max_group.sort(key=lambda x: x.d)
    group_d = []

    black2 = np.zeros((height, width, depth), np.uint8)
    for i in range(0, len(max_group)):
        ls = max_group[i]
        x1, x2, y1, y2 = ls.x1, ls.x2, ls.y1, ls.y2
        cv2.line(black2, (x1, y1), (x2, y2), (0, 0, 255), 3)

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

    line_min = min(max_group, key=lambda x: x.d)
    line_max = max(max_group, key=lambda x: x.d)

    x_avg = sum([ls.x for ls in max_group]) / (1.0 * len(max_group))
    y_avg = sum([ls.y for ls in max_group]) / (1.0 * len(max_group))
    xx = x_avg + 100
    yy = y_avg + 100 * math.tan(line_min.theta)

    new_line = LineSegment(x_avg, y_avg, xx, yy)
    new_line.update(line_min.theta)
    delta_d = (line_min.d + line_max.d) / 2.0 - new_line.d

    x_avg += delta_d * math.cos(line_min.theta + math.pi / 2.0)
    y_avg += delta_d * math.sin(line_min.theta + math.pi / 2.0)

    cv2.circle(black, (int(x_avg), int(y_avg)), 30, (255, 255, 255), -1)

    w = line_max.distance(line_min.x, line_min.y)
    h = w / 3.0 * 2.4
    r_2 = (w ** 2 + h ** 2) / 4.0

    l = line_min
    x1, x2, y1, y2 = l.x1, l.x2, l.y1, l.y2
    a = (y1 - y2) ** 2 + (x1 - x2) ** 2
    b = 2.0 * ((y1 - y2) * (y1 - y_avg) + (x1 - x2) * (x1 - x_avg))
    c = (y1 - y_avg) ** 2 + (x1 - x_avg) ** 2 - r_2

    t1 = (-b - math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
    t2 = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

    xn1 = x1 + t1 * (x1 - x2)
    yn1 = y1 + t1 * (y1 - y2)

    xn2 = x1 + t2 * (x1 - x2)
    yn2 = y1 + t2 * (y1 - y2)

    l = line_max
    x1, x2, y1, y2 = l.x1, l.x2, l.y1, l.y2
    a = (y1 - y2) ** 2 + (x1 - x2) ** 2
    b = 2.0 * ((y1 - y2) * (y1 - y_avg) + (x1 - x2) * (x1 - x_avg))
    c = (y1 - y_avg) ** 2 + (x1 - x_avg) ** 2 - r_2

    t1 = (-b - math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
    t2 = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

    xn3 = x1 + t1 * (x1 - x2)
    yn3 = y1 + t1 * (y1 - y2)

    xn4 = x1 + t2 * (x1 - x2)
    yn4 = y1 + t2 * (y1 - y2)

    x_max = max_group[0].x1
    x_min = max_group[len(max_group) - 1].x1

    y_min = min(map(lambda l: l.y1, max_group))
    y_max = max(map(lambda l: l.y2, max_group))

    cv2.circle(black, (int(xn1), int(yn1)), 30, (255, 255, 255), -1)
    cv2.circle(black, (int(xn2), int(yn2)), 30, (255, 255, 255), -1)
    cv2.circle(black, (int(xn3), int(yn3)), 30, (255, 255, 255), -1)
    cv2.circle(black, (int(xn4), int(yn4)), 30, (255, 255, 255), -1)
    # 3 4
    # 1 2
    pts1 = np.float32([[xn1, yn1], [xn3, yn3], [xn2, yn2], [xn4, yn4]])
    # pts1 = np.float32([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 200], [300, 200]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(original, M, (300, 200))

    xn1, yn1, xn3, yn3, xn2, yn2, xn4, yn4 = int(xn1), int(yn1), int(xn3), int(yn3), int(xn2), int(yn2), int(xn4), int(
        yn4)

    cv2.line(img, (xn1, yn1), (xn2, yn2), yellow, thickness)
    cv2.line(img, (xn2, yn2), (xn4, yn4), yellow, thickness)
    cv2.line(img, (xn4, yn4), (xn3, yn3), yellow, thickness)
    cv2.line(img, (xn1, yn1), (xn3, yn3), yellow, thickness)

    cv2.line(black, (xn1, yn1), (xn2, yn2), yellow, thickness)
    cv2.line(black, (xn2, yn2), (xn4, yn4), yellow, thickness)
    cv2.line(black, (xn4, yn4), (xn3, yn3), yellow, thickness)
    cv2.line(black, (xn1, yn1), (xn3, yn3), yellow, thickness)

    cv2.imwrite(output_path + 'line_' + filename, img)
    cv2.imwrite(output_path + 'only_line_' + filename, black)
    cv2.imwrite(output_path + 'dst_' + filename, dst)
cv2.destroyAllWindows()
