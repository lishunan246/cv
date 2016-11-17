# coding=utf-8

import numpy as np
import cv2
import math
import os
from LineSegment import LineSegment, LineSegmentGroup
from scipy import signal

# 条形码比例3:2
theta_tolerance = 0.2
DST_WIDTH = 300
DST_HEIGHT = 200
yellow = (0, 255, 255)
green = (0, 255, 0)
red = (0, 0, 255)
thickness = 10
font = cv2.FONT_HERSHEY_SIMPLEX
input_path = 'input//'
output_path = 'output\\'
test_path = 'test//'

input_dic = {}
test_dic = {}


def get_grays(filename):
    original = cv2.imread(filename, cv2.IMREAD_COLOR)

    filename = filename.split('//')[1]
    img = original.copy()
    height, width, depth = img.shape

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, th / 2, th)
    # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=80, maxLineGap=10)
    if lines is None:
        print(filename + ' No line detected.')
        return

    lss = []
    black = np.zeros((height, width, depth), np.uint8)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), green, 1)
        cv2.line(black, (x1, y1), (x2, y2), green, 1)
        ls = LineSegment(x1, y1, x2, y2)
        lss.append(ls)

    lines = sorted(lss, key=lambda x: x.theta)
    line_group = []

    for ls in lines:

        if len(line_group) == 0:  # first line group
            line_group.append([ls])
        else:
            done = False
            for i in range(0, len(line_group)):
                if abs(line_group[i][0].theta - ls.theta) < theta_tolerance:  # 比较斜率
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
        cv2.line(black2, (x1, y1), (x2, y2), red, 3)

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

    x_avgt = sum([ls.x for ls in max_group]) / (1.0 * len(max_group))
    y_avgt = sum([ls.y for ls in max_group]) / (1.0 * len(max_group))
    xx = x_avgt + 100
    yy = y_avgt + 100 * math.tan(avg_theta)

    new_line = LineSegment(x_avgt, y_avgt, xx, yy)
    new_line.update(avg_theta)
    delta_d = (line_min.d + line_max.d) / 2.0 - new_line.d
    # cv2.circle(black, (int(x_avgt), int(y_avgt)), 20, (255, 0, 255), -1)
    xn = np.zeros((4, 2))
    yn = np.zeros((4, 2))
    dn = np.zeros(2)

    for i in range(0, 2):
        try:
            if i == 0:
                x_avg = x_avgt + delta_d * math.cos(avg_theta + math.pi / 2.0)
                y_avg = y_avgt + delta_d * math.sin(avg_theta + math.pi / 2.0)
                enlarge = (line_max.d - line_min.d) / 8.0
            else:
                x_avg = x_avgt - delta_d * math.cos(avg_theta + math.pi / 2.0)
                y_avg = y_avgt - delta_d * math.sin(avg_theta + math.pi / 2.0)
                enlarge = (line_max.d - line_min.d) / 8.0

            # print str(x_avg) + ' ' + str(y_avg)
            cv2.circle(black, (int(x_avg), int(y_avg)), 10, (255, 255, 255), -1)

            cv2.putText(black, 'theta: ' + str(avg_theta) + 'd: ' + str(delta_d), (80, 300), font, 1, (255, 255, 255),
                        2)
            cv2.imwrite(output_path + 'theta_' + filename, black)
            # continue

            enlarge = 0
            x1, y1, x2, y2 = line_min.x1, line_min.y1, line_min.x2, line_min.y2
            x1 -= enlarge * math.cos(avg_theta + math.pi / 2.0)
            x2 -= enlarge * math.cos(avg_theta + math.pi / 2.0)
            y1 -= enlarge * math.sin(avg_theta + math.pi / 2.0)
            y2 -= enlarge * math.sin(avg_theta + math.pi / 2.0)

            line_min_new = LineSegment(x1, y1, x2, y2)

            x1, y1, x2, y2 = line_max.x1, line_max.y1, line_max.x2, line_max.y2
            x1 += enlarge * math.cos(avg_theta + math.pi / 2.0)
            x2 += enlarge * math.cos(avg_theta + math.pi / 2.0)
            y1 += enlarge * math.sin(avg_theta + math.pi / 2.0)
            y2 += enlarge * math.sin(avg_theta + math.pi / 2.0)

            line_max_new = LineSegment(x1, y1, x2, y2)

            w = line_max_new.distance(line_min_new.x, line_min_new.y)
            h = w / 3.0 * 2.1
            r_2 = (w ** 2 + h ** 2) / 4.0

            l = line_min_new
            x1, x2, y1, y2 = l.x1, l.x2, l.y1, l.y2
            a = (y1 - y2) ** 2 + (x1 - x2) ** 2
            b = 2.0 * ((y1 - y2) * (y1 - y_avg) + (x1 - x2) * (x1 - x_avg))
            c = (y1 - y_avg) ** 2 + (x1 - x_avg) ** 2 - r_2

            t1 = (-b - math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
            t2 = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

            xn[0, i] = x1 + t1 * (x1 - x2)
            yn[0, i] = y1 + t1 * (y1 - y2)

            xn[1, i] = x1 + t2 * (x1 - x2)
            yn[1, i] = y1 + t2 * (y1 - y2)

            l = line_max_new
            x1, x2, y1, y2 = l.x1, l.x2, l.y1, l.y2
            a = (y1 - y2) ** 2 + (x1 - x2) ** 2
            b = 2.0 * ((y1 - y2) * (y1 - y_avg) + (x1 - x2) * (x1 - x_avg))
            c = (y1 - y_avg) ** 2 + (x1 - x_avg) ** 2 - r_2

            t1 = (-b - math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
            t2 = (-b + math.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)

            xn[2, i] = x1 + t1 * (x1 - x2)
            yn[2, i] = y1 + t1 * (y1 - y2)

            xn[3, i] = x1 + t2 * (x1 - x2)
            yn[3, i] = y1 + t2 * (y1 - y2)

            dn[i] = abs(math.sqrt((xn[0, i] - xn[1, i]) ** 2 + (yn[0, i] - yn[1, i]) ** 2) - math.sqrt(
                (xn[2, i] - xn[3, i]) ** 2 + (yn[2, i] - yn[3, i]) ** 2))

        except ValueError:
            dn[i] = float('inf')

    ni = 1 if dn[1] < dn[0] else 0

    xn = xn[:, ni]
    yn = yn[:, ni]

    for i in range(0, 4):
        cv2.circle(black, (int(xn[i]), int(yn[i])), 30, (255, 255, 255), -1)

    # 4 3
    #  2 1
    pts1 = np.float32([[xn[1], yn[1]], [xn[3], yn[3]], [xn[0], yn[0]], [xn[2], yn[2]]])
    pts2 = np.float32([[0, 0], [DST_WIDTH, 0], [0, DST_HEIGHT], [DST_WIDTH, DST_HEIGHT]])
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(original, M, (DST_WIDTH, DST_HEIGHT))

    xn = np.uint(xn)
    yn = np.uint(yn)
    cv2.line(img, (xn[0], yn[0]), (xn[1], yn[1]), yellow, thickness)
    cv2.line(img, (xn[1], yn[1]), (xn[3], yn[3]), yellow, thickness)
    cv2.line(img, (xn[3], yn[3]), (xn[2], yn[2]), yellow, thickness)
    cv2.line(img, (xn[2], yn[2]), (xn[0], yn[0]), yellow, thickness)

    cv2.line(black, (xn[0], yn[0]), (xn[1], yn[1]), yellow, thickness)
    cv2.line(black, (xn[1], yn[1]), (xn[3], yn[3]), yellow, thickness)
    cv2.line(black, (xn[3], yn[3]), (xn[2], yn[2]), yellow, thickness)
    cv2.line(black, (xn[2], yn[2]), (xn[0], yn[0]), yellow, thickness)

    cv2.imwrite(output_path + 'line_' + filename, img)
    cv2.imwrite(output_path + 'only_line_' + filename, black)

    black_small = np.zeros(dst.shape, np.uint8)
    gray_small = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # 检测数字位置
    dst1 = cv2.cornerHarris(gray_small, 2, 3, 0.04)
    dst1 = cv2.dilate(dst1, None)
    t = dst1 > 0.01 * dst1.max()
    black_small[t] = [0, 0, 255]
    up = t[0:40, :]
    down = t[DST_HEIGHT - 41:DST_HEIGHT - 1, :]
    if sum(sum(up)) > sum(sum(down)):
        # 旋转180度
        dst = cv2.flip(dst, -1)
        gray_small = cv2.flip(gray_small, -1)

    # cv2.imwrite(output_path + 's_only_line_' + filename, black_small)
    cv2.imwrite(output_path + 'dst_' + filename, dst)

    # 使用平均灰度值进行二值化
    height, width = gray_small.shape
    avg_gray = sum(sum(1.0 * gray_small)) / (height * width)

    cv2.imwrite(output_path + "gray_" + filename, gray_small)
    std = np.std(gray_small)
    gray_small = (gray_small - avg_gray) / std
    return gray_small

for file in os.listdir(input_path):
    input_dic[file] = get_grays(input_path + file)

for file in os.listdir(test_path):
    test_dic[file] = get_grays(test_path + file)

for test_ in test_dic.keys():
    if test_dic[test_] is None:
        print(test_ + ' fail')
        continue

    t_max = 0
    name = ''
    for input_ in input_dic.keys():
        t = signal.fftconvolve(input_dic[input_], cv2.flip(test_dic[test_], -1), mode='same')
        if t_max < t.max():
            t_max = t.max()
            name = input_

    print(test_ + ' like ' + name + ' ' + str(t_max / (DST_HEIGHT * DST_WIDTH)))

