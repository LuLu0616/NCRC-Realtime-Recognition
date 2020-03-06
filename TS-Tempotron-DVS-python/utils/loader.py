# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-29 20-46-27
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-29 21:30:09
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  :
'''

import scipy.io as io


class AER:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.p = 0
        self.t = 0


class Layer:
    def __init__(self, radius, tau, num_feature, C, count, alpha, beta, pk, image_size):
        self.radius = radius
        self.tau = tau
        self.num_feature = num_feature
        self.C = C
        self.count = count
        self.alpha = alpha
        self.beta = beta
        self.pk = pk
        self.image_size = image_size

# def read_dataset(file):                     # N-MNIST
#     datafile = open(file, 'rb')
#     raw_data = np.fromfile(datafile, dtype=np.uint8)
#     raw_data = np.uint32(raw_data)
#     datafile.close()
#     # print("raw_data.size =", raw_data.size)
#     TD = AER()
#     TD.x = raw_data[0::5]
#     TD.y = raw_data[1::5]
#     TD.p = (raw_data[2::5] & 128) >> 7
#     TD.t = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
#     TD.x = TD.x.tolist()
#     TD.y = TD.y.tolist()
#     TD.p = TD.p.tolist()
#     TD.t = TD.t.tolist()
#     # print(TD.x, TD.y, TD.p, TD.t)
#     return TD


def read_simple(file):             # MNIST-DVS
    data = io.loadmat(file, squeeze_me = True, struct_as_record = False)
    TD = AER()
    # print(data['TD'])
    TD.x = data['TD'].x
    TD.y = data['TD'].y
    TD.p = data['TD'].p
    TD.t = data['TD'].ts
    TD.x = TD.x.tolist()
    TD.y = TD.y.tolist()
    TD.p = TD.p.tolist()
    TD.t = TD.t.tolist()
    # print(TD.x, TD.y, TD.p, TD.t)
    return TD


def read_timesurface_params(file, LAYER):
    data = io.loadmat(file)
    radius = data['params'][LAYER][0][0][0][0][0][0][0]
    tau = data['params'][LAYER][0][0][0][0][1][0][0]
    num_feature = data['params'][LAYER][0][0][0][0][2][0][0]
    C = data['params'][LAYER][0][0][0][0][3]
    count = data['params'][LAYER][0][0][0][0][4][0][0]
    alpha = data['params'][LAYER][0][0][0][0][5][0][0]
    beta = data['params'][LAYER][0][0][0][0][6][0][0]
    pk = data['params'][LAYER][0][0][0][0][7]
    image_size = data['params'][LAYER][0][0][0][0][8][0]
    # print(radius, tau, num_feature, C, count, alpha, beta, pk, image_size)
    layer = Layer(radius=radius, tau=tau, num_feature=num_feature, C=C, count=count, alpha=alpha, beta=beta, pk=pk, image_size=image_size)
    return layer


def weightloader(file):
    data = io.loadmat(file)
    weights = data['TrainedWt']
    acc_train = data['correctRate']
    return weights, acc_train