from numba.experimental import jitclass
import numba as nb 

c_segment = [
    ("x_min",nb.int32),
    ("x_max",nb.int32),
    ("y_min",nb.int32),
    ("y_max",nb.int32),
    ("tag",nb.int32),
    ("size",nb.int32)
]

@jitclass(c_segment)
class Segment:
    def __init__(self,x_min,x_max,y_min,y_max,tag):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.tag=tag
        self.size=0
    def update(self,x,y):
        self.x_min = min(self.x_min,x)
        self.x_max = max(self.x_max,x)
        self.y_min = min(self.y_min,y)
        self.y_max = max(self.y_max,y)
        self.size += 1
    def ontop(self,other):
        horizontally_alligned = self.y_min + 55 > other.y_min and self.y_max < 55 + other.y_max and self.y_min < other.y_max and other.y_min < self.y_max
        vertically_alligned = self.x_max + 20 > other.x_min and self.x_max < 20 + other.x_min
        bottom_big = self.size < other.size
        bottom_long = (self.x_max-self.x_min)/2 < other.x_max-other.x_min
        top_proportional = self.x_max-self.x_min < (self.y_max-self.y_min)*2.5
        return horizontally_alligned and vertically_alligned and bottom_big and bottom_long and top_proportional
    def __str__(self):
        return "x=({}-{}) y=({}-{}) size = {} tag = {}".format(self.y_min,self.y_max,self.x_min,self.x_max,self.size,self.tag)
    __repr__=__str__

