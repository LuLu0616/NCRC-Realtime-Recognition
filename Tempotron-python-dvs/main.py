# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-27 19:22:28
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-28 14:29:46
   @FilePath     : \WorkSpace\Local\Tempotron-python\main.py
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Test EventDrivenTempotron
'''
import os
import scipy.io as io
from modules.tempotron import Tempotron

root_path = './Local/Tempotron-python/'
data_path = os.path.join(root_path, 'data')
params_path = os.path.join(root_path, 'params')
for path in [data_path, params_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def weightloader(file):
    data = io.loadmat(file)
    weights = data['TrainedWt']
    acc_train = data['correctRate']
    return weights, acc_train


if __name__ == "__main__":
    # for root, dirs, files in os.walk(data_path):
    data = io.loadmat(os.path.join(data_path, 'PtnCell_spk.mat'),
                      squeeze_me=True, struct_as_record=False)
    weights, acc_train = weightloader(os.path.join(params_path, 'TrainedWt.mat'))
    obj = Tempotron(weights, IsTraining=0, dataloader=data, maxEpoch=1)
    obj.eventdriven()