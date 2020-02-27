import numpy as np
from parameters import *


class Coding:
    def __init__(self, TD):
        self.AllVec = np.zeros((len(TD.t), 1))
        self.AllAddr = np.zeros((len(TD.t), 1))
        self.size_x = imsize2[0]
        self.size_y = imsize2[1]
        for i in range(len(TD.t)):
            self.AllVec[i] = TD.t[i] / 1e3
            self.AllAddr[i] = TD.p[i] * self.size_x * self.size_y + (TD.x[i] - 1) * self.size_y + TD.y[i]
        self.maxT = max(self.AllVec)
        for i in range(len(self.AllVec)):
            self.AllVec[i] = math.ceil(255 / self.maxT * self.AllVec[i])
        # self.AllVec = [math.ceil(255 / self.maxT * i) for i in self.AllVec]
