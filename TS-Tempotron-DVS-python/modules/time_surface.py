# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-29 20:37:11
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-29 21:52:30
   @FilePath     : \WorkSpace\Local\TS-Tempotron-DVS-python\modules\time_surface.py
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''
import os
import numpy as np
import math
from utils.loader import read_timesurface_params

root_path = './'
params_path = os.path.join(root_path, 'params')
STSF_param = os.path.join(params_path, "STSF_params.mat")
layer1 = read_timesurface_params(STSF_param, "layer1")           # realtest
layer2 = read_timesurface_params(STSF_param, "layer2")           # realtest


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
    t_last = np.ones((size_y + 2 * radius + 1, size_x + 2 * radius + 1, max(TD.p) + 1)) * tau * (-1)
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
                    S[radius + ry, radius + rx] = lut[lut_addr]
        t_last[yi, xi, pi] = ti
        if S.sum() == 0:
            output_index = num_feature
        else:
            min_distance = float("inf")
            for index in range(num_feature):
                temp = C[:, :, index] - S
                distance = np.linalg.norm(temp)
                if distance < min_distance:
                    min_distance = distance
                    output_index = index
        feat.x[i] = TD.x[i]
        feat.y[i] = TD.y[i]
        feat.t[i] = TD.t[i]
        feat.p[i] = output_index
    feat = RemoveNulls(feat, num_feature)
    # print(len(feat.t))
    if ispool == 1:
        feat.x = [math.ceil(i / pooling_extent) for i in feat.x]
        feat.y = [math.ceil(i / pooling_extent) for i in feat.y]
        feat = ImplementRefraction(feat, refractory_period)
        # print(len(feat.t))
    return feat
