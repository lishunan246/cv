# coding=utf-8
import numpy
import math


class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1  # y1在下 y2在上

        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.k = (y1 - y2) / (1.0 * (x1 - x2)) if x1 != x2 else float('nan')
        self.b = y1 - self.k * x1
        self.theta = math.atan2(y1-y2,x1-x2)
        self.d = -self.b / self.k * numpy.sin(self.theta) if self.k != 0 else x1  # 到直线到原点的最近距离
    def update(self,theta):
        self.theta=theta
        x=(self.x1+self.x2)/2.0
        y=(self.y1+self.y2)/2.0
        #self.d=numpy.sqrt(x*x+y*y)*numpy.sin(self.theta)
        self.d=x

class LineSegmentGroup:
    def __init__(self):
        self.d_max = None
        self.d_min = None
        self.lines = []
        self.D = 65

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
        return distance < self.D
