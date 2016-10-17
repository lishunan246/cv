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
        self.theta = numpy.arctan(self.k) if x1 != x2 else math.pi / 2
