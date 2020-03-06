# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-29 20:34:52
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-29 21:10:49
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add description
'''
import math
import numpy as np
imsize0 = [29, 29]
poolsize1 = 2
poolsize2 = 2
imsize1 = [math.ceil(i / poolsize1) for i in imsize0]
imsize2 = [math.ceil(i / poolsize2) for i in imsize1]


class Coding:
    def __init__(self, TD):
        self.AllVec = np.zeros((len(TD.t), 1))
        self.AllAddr = np.zeros((len(TD.t), 1))
        self.size_x = imsize2[0]
        self.size_y = imsize2[1]
        for i in range(len(TD.t)):
            self.AllVec[i] = TD.t[i] / 1e3
            self.AllAddr[i] = TD.p[i] * self.size_x * \
                self.size_y + (TD.x[i] - 1) * self.size_y + TD.y[i]
        self.maxT = 2.861772000000000e+03
        for i in range(len(self.AllVec)):
            self.AllVec[i] = math.ceil(255 / self.maxT * self.AllVec[i])
        # self.AllVec = [math.ceil(255 / self.maxT * i) for i in self.AllVec]
