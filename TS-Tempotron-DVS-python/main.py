# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-27 19:22:28
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-29 22:38:17
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Test TS+EventDrivenTempotron
'''

import os
import time
# import scipy.io as io
from modules.tempotron import Tempotron
from modules.time_surface import D_timesurface_realtime
from utils.encode import Coding
import utils.loader as ld

# parameters
ispool1 = 1
ispool2 = 1
poolsize1 = 2
poolsize2 = 2

root_path = './' ###TS and Weights
data_root = 'D:\Projects\matlab\DVS-Realtime-Recognition-matlab\perGestureOut'  ### dataset
params_path = os.path.join(root_path, 'params')
for path in [params_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    CorrectNum = 0
    Acc = 0
    step = 0
    for root, dirs, files in os.walk(data_root):
        for file in files:
            label = int(root[-1:])
            TD = ld.read_simple(os.path.join(root, file))

            ts_start = time.time()
            if min(TD.x) == 0 or min(TD.y) == 0:
                TD.x = [i + 1 for i in TD.x]
                TD.y = [i + 1 for i in TD.y]
            TD.t = [i - TD.t[0] for i in TD.t]
            STSF_param = os.path.join(params_path, "STSF_params.mat")
            layer1 = ld.read_timesurface_params(STSF_param, "layer1")  # realtest
            layer2 = ld.read_timesurface_params(STSF_param, "layer2")  # realtest
            # print(len(TD.t))
            TD = D_timesurface_realtime(TD, 1, ispool1, poolsize1, 5e3)  # 10927
            TD = D_timesurface_realtime(TD, 2, ispool2, poolsize2, 5e3)  # 7055
            # print(min(TD.x), min(TD.y))
            PtnCell = Coding(TD)
            print('TS Time elasped per image: %.2fs' % (time.time() - ts_start))
            # weights = np.random.randn(nAfferents, noutputs, nNeuronPerOutput)

            tp_start = time.time()
            weights, acc_train = ld.weightloader(os.path.join(params_path, 'TrainedWt.mat'))
            obj = Tempotron(weights, data=PtnCell)
            pred = obj.eventdriven()
            if label == pred:
                CorrectNum += 1
            Acc = CorrectNum / (step + 1)
            print('Tempotron Time elasped per image: %.2fs' % (time.time() - tp_start))
            print(
                'Iter: [%d/%d], Acc: %.2f, Time elasped: %.2fs' % (step + 1, 10000, Acc * 100, time.time() - ts_start))
            start_time = time.time()
            step += 1
