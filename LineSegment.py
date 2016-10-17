class LineSegment:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.k = (y1 - y2) / (1.0*(x1 - x2))
        self.b = y1 - self.k * x1
