# coding=utf-8
import numpy
import math
from decimal import *


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1  # y1在下 y2在上

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.k = (y1 - y2) / (1.0 * (x1 - x2)) if x1 != x2 else float('nan')
        self.theta = math.atan2(y1 - y2, x1 - x2)

        self.b = y1 - self.k * x1 if x1 != x2 else float('nan')

        # 到直线到原点的最近距离
        #self.d = -self.b * (numpy.sin(self.theta) / math.tan(self.theta)) if not math.isnan(self.k) else x1

    def update(self, theta):
        old_theta = self.theta
        self.theta = theta
        x = (self.x1 + self.x2) / 2.0
        y = (self.y1 + self.y2) / 2.0

        d_theta = self.theta - old_theta

        p1 = numpy.array([self.x1, self.y1, 1])
        p2 = numpy.array([self.x2, self.y2, 1])
        translate1 = numpy.array([
            [1, 0, 0],
            [0, 1, 0],
            [x - self.x1, y - self.y1, 1]
        ])

        translate2 = numpy.array([
            [1, 0, 0],
            [0, 1, 0],
            [x - self.x2, y - self.y2, 1]
        ])
        p1 = p1.dot(translate1)
        p2 = p2.dot(translate2)
        rotate = numpy.array([
            [numpy.cos(d_theta), numpy.sin(d_theta), 0],
            [-numpy.sin(d_theta), numpy.cos(d_theta), 0],
            [0, 0, 1]
        ])
        p1 = p1.dot(rotate).dot(numpy.array([
            [1, 0, 0],
            [0, 1, 0],
            [self.x1 - x, self.y1 - y, 1]
        ]))

        p2 = p2.dot(rotate).dot(numpy.array([
            [1, 0, 0],
            [0, 1, 0],
            [self.x2 - x, self.y2 - y, 1]
        ]))
        self.x1,self.y1 ,t = p1.astype(int)
        self.x2, self.y2, t = p2.astype(int)

        # self.d = Decimal(y) - Decimal(math.tan(theta)) * Decimal(x) if theta != math.pi / 2 else x
        p1 = (numpy.array([[self.x1, self.y1]])).astype(float)
        p2 = numpy.array([[self.x2, self.y2]]).astype(float)
        p0 = numpy.array([[0, 0]]).astype(float)

        self.d = (numpy.cross(p0 - p1, p0 - p2)**2).sum() / (1.0 * ((p1 - p2)**2) .sum())
        #self.d = x
        self.d=numpy.sqrt(self.d)

class LineSegmentGroup:
    def __init__(self):
        self.d_max = None
        self.d_min = None
        self.lines = []
        self.D = 80

    def insert(self, ls):
        if self.d_max is None:
            self.d_max = ls.d
            self.d_min = ls.d
        elif self.d_min > ls.d:
            self.d_min = ls.d
        elif self.d_max < ls.d:
            self.d_max = ls.d

        self.lines.append(ls)

    def is_near(self, ls):
        distance = min(abs(ls.d - self.d_min), abs(ls.d - self.d_max))
        if distance < self.D:
            return True
        else:
            #print distance, ls.d
            return False
