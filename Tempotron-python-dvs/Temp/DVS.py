import numpy as np
import random
import scipy.io as scio
from parameters import *
from encode import *
from EventDrivenTempotron import *


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


'''def read_dataset(file):                     # N-MNIST
    datafile = open(file, 'rb')
    raw_data = np.fromfile(datafile, dtype=np.uint8)
    raw_data = np.uint32(raw_data)
    datafile.close()
    # print("raw_data.size =", raw_data.size)
    TD = AER()
    TD.x = raw_data[0::5]
    TD.y = raw_data[1::5]
    TD.p = (raw_data[2::5] & 128) >> 7
    TD.t = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    TD.x = TD.x.tolist()
    TD.y = TD.y.tolist()
    TD.p = TD.p.tolist()
    TD.t = TD.t.tolist()
    # print(TD.x, TD.y, TD.p, TD.t)
    return TD
'''


def read_dataset(file):             # MNIST-DVS

    data = scio.loadmat(file)
    TD = AER()
    # print(data['TD'])
    TD.x = data['TD']['x'][0][0][0]
    TD.y = data['TD']['y'][0][0][0]
    TD.p = data['TD']['p'][0][0][0]
    TD.t = data['TD']['ts'][0][0][0]
    TD.x = TD.x.tolist()
    TD.y = TD.y.tolist()
    TD.p = TD.p.tolist()
    TD.t = TD.t.tolist()
    # print(TD.x, TD.y, TD.p, TD.t)
    return TD


def read_timesurface_params(file, LAYER):
    data = scio.loadmat(file)
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


def read_tempotron_weights(file):
    data = scio.loadmat(file)
    weights = data['TrainedWt']
    correctRate = data['correctRate']
    return weights


def RemoveNulls(feat, num_feature):
    # print(type(feat.x), type(feat.y), type(feat.p), type(feat.t))
    if num_feature == 0:
        index = [i for i, d in enumerate(feat.t) if d == num_feature]
    else:
        index = [i for i, d in enumerate(feat.p) if d == num_feature]
    for i in reversed(index):
        feat.x.pop(i)
        feat.y.pop(i)
        feat.p.pop(i)
        feat.t.pop(i)
    # print(len(feat.x), len(feat.y), len(feat.p), len(feat.t))
    return feat


def ImplementRefraction(feat, refractory_period):
    feat.t = [i + refractory_period for i in feat.t]
    print(len(feat.t))
    LastTime = np.zeros((max(feat.x) + 1, max(feat.y) + 1))
    for i in range(len(feat.t)):
        if feat.t[i] - LastTime[feat.x[i], feat.y[i]] > refractory_period:
            LastTime[feat.x[i], feat.y[i]] = feat.t[i]
        else:
            feat.t[i] = 0
    feat = RemoveNulls(feat, 0)
    feat.t = [i - refractory_period for i in feat.t]
    return feat


def D_timesurface(TD, layer, ispool, pooling_extent, refractory_period):
    global layer1, layer2
    if layer == 1:
        param = layer1
    else:
        param = layer2
    radius = param.radius
    tau = param.tau
    num_feature = param.num_feature
    C = param.C
    count = param.count
    alpha = param.alpha
    beta = param.beta
    pk = param.pk
    image_size = param.image_size
    size_x = image_size[0]
    size_y = image_size[1]
    dt = 10
    t = np.arange(0, tau + dt, dt)
    lut = [math.exp(-i / tau) for i in t]
    t_last = np.ones((size_y + 2 * radius, size_x + 2 * radius, max(TD.p) + 1)) * (-tau)
    feat = TD
    for i in range(len(TD.t)):
        xi = TD.x[i] + radius
        yi = TD.y[i] + radius
        ti = TD.t[i]
        pi = TD.p[i]
        S = np.zeros((2 * radius + 1, 2 * radius + 1))
        for rx in range(-radius, radius):
            for ry in range(-radius, radius):
                delta_t = ti - t_last[yi + ry, xi + rx, pi]
                if delta_t < tau:
                    lut_addr = int(round(delta_t / dt))
                    S[radius + rx, radius + ry] = lut[lut_addr]
        t_last[yi, xi, pi] = ti
        if S.sum() == 0:
            output_index = num_feature
        else:
            if count < num_feature:
                C[:, :, count] = S
                output_index = count
                count += 1
            else:
                min_distance = float("inf")
                for index in range(num_feature):
                    temp = C[:, :, index] - S
                    distance = np.linalg.norm(temp)
                    if distance < min_distance:
                        distance = min_distance
                        output_index = index
        feat.x[i] = TD.x[i]
        feat.y[i] = TD.y[i]
        feat.t[i] = TD.t[i]
        feat.p[i] = output_index
        if output_index < num_feature & count == num_feature:
            Ck = C[:, :, output_index]
            Ck = Ck[:]
            Sk = S[:]
            norm_Ck = np.linalg.norm(Ck)
            norm_Sk = np.linalg.norm(Sk)
            if norm_Ck == 0 or norm_Sk == 0:
                beta = 0
            else:
                sum = 0
                for a in range(2 * radius + 1):
                    for b in range(2 * radius + 1):
                        sum += Ck[a, b] * Sk[a, b]
                beta = sum / (norm_Sk * norm_Ck)
            alpha = 0.01 / (1 + pk[output_index] / 20000)
            t = [i * beta for i in C[:, :, output_index]]
            t = S - t
            t = [i * alpha for i in t]
            C[:, :, output_index] += t
            pk[output_index] += 1
    feat = RemoveNulls(feat, num_feature)
    param.C = C
    param.count = count
    param.alpha = alpha
    param.beta = beta
    param.pk = pk
    if layer == 1:
        layer1 = param
    else:
        layer2 = param
    if ispool == 1:
        feat.x = [math.ceil(i / pooling_extent) for i in feat.x]
        feat.y = [math.ceil(i / pooling_extent) for i in feat.y]
        feat = ImplementRefraction(feat, refractory_period)
    return feat


def D_timesurface_realtime(TD, layer, ispool, pooling_extent, refractory_period):
    global layer1, layer2
    if layer == 1:
        param = layer1
    else:
        param = layer2
    radius = int(param.radius)
    tau = param.tau
    num_feature = param.num_feature
    C = param.C
    image_size = param.image_size
    size_x = image_size[0]
    size_y = image_size[1]
    dt = 10
    t = np.arange(0, tau + dt, dt)
    lut = [math.exp(-1 * i / tau) for i in t]
    t_last = np.ones((size_y + 2 * radius, size_x + 2 * radius, max(TD.p) + 1)) * tau * (-1)
    feat = TD
    for i in range(len(TD.t)):
        xi = TD.x[i] + radius
        yi = TD.y[i] + radius
        ti = TD.t[i]
        pi = TD.p[i]
        S = np.zeros((2 * radius + 1, 2 * radius + 1))
        for rx in range(-radius, radius + 1):
            for ry in range(-radius, radius + 1):
                delta_t = ti - t_last[yi + ry, xi + rx, pi]
                # print(ti, delta_t)
                if delta_t < tau:
                    lut_addr = int(round(delta_t / dt))
                    # print(radius+rx, radius+ry, lut_addr)
                    S[radius + rx, radius + ry] = lut[lut_addr]
        t_last[yi, xi, pi] = ti
        if S.sum() == 0:
            output_index = num_feature
        else:
            min_distance = float("inf")
            for index in range(num_feature):
                temp = C[:, :, index] - S
                distance = np.linalg.norm(temp)
                if distance < min_distance:
                    distance = min_distance
                    output_index = index
        feat.x[i] = TD.x[i]
        feat.y[i] = TD.y[i]
        feat.t[i] = TD.t[i]
        feat.p[i] = output_index
    feat = RemoveNulls(feat, num_feature)
    if ispool == 1:
        feat.x = [math.ceil(i / pooling_extent) for i in feat.x]
        feat.y = [math.ceil(i / pooling_extent) for i in feat.y]
        feat = ImplementRefraction(feat, refractory_period)
    return feat


if __name__ == '__main__':
    # layer1 = Layer(radius=2, tau=20e3, num_feature=6, C=np.zeros((5, 5, 6)), count=0, alpha=0, beta=0, pk=np.ones((6, 1)), image_size=imsize0)          # train
    # layer2 = Layer(radius=4, tau=200e3, num_feature=18, C=np.zeros((9, 9, 18)), count=0, alpha=0, beta=0, pk=np.ones((18, 1)), image_size=imsize1)      # train
    TD = read_dataset("./local_use/dvs-python/MNIST_DVS_full_0_1000.mat")          # TD: 一段事件流
    if reRunAll | reGetFeature:
        if min(TD.x) == 0 | min(TD.y) == 0:
            TD.x = [i + 1 for i in TD.x]
            TD.y = [i + 1 for i in TD.y]
        layer1 = read_timesurface_params("./local_use/dvs-python/STSF_params.mat", "layer1")           # realtest
        layer2 = read_timesurface_params("./local_use/dvs-python/STSF_params.mat", "layer2")           # realtest
        # print(len(TD.t))
        TD = D_timesurface_realtime(TD, 1, ispool1, poolsize1, 5e3)
        TD = D_timesurface_realtime(TD, 2, ispool2, poolsize2, 5e3)
        # print(min(TD.x), min(TD.y))
        print("Feature Extraction End...")
    if reRunAll | reGenPtnCell:
        PtnCell = [0] * 100
        PtnCell[0] = Coding(TD)
        print("Encoding End...")
    # for i in range(len(TD.t)):
    #     print(TD.x[i], TD.y[i], TD.p[i], TD.t[i], PtnCell[0].AllVec[i], PtnCell[0].AllAddr[i])

    if reRunAll | reRealTimeTest:
        weights = read_tempotron_weights("TrainedWt.mat")
        # weights = np.random.randn(nAfferents, noutputs, nNeuronPerOutput)
        obj = Tempotron(weights=weights, IsTraining=0, PtnCell=PtnCell, maxEpoch=1, lmd=lmd)
        obj.EventDrivenTempotron()
        print("Realtime Testing End...")
