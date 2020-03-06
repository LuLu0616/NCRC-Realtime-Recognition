# coding=utf-8
'''
   @Author       : Noah
   @Version      : v1.0.0
   @Date         : 2020-02-27 21:06:34
   @LastEditors: Please set LastEditors
   @LastEditTime: 2020-02-29 22:31:28
   @FilePath     : \WorkSpace\Local\Tempotron-python\modules\tempotron.py
   @CopyRight (c): 2019 NCRC, SCU. All rights reserved.
   @Description  : Please add descriptioon
'''
import numpy as np


class Tempotron:
    def __init__(self, weights, data):
        self.weights = weights
        # print(self.weights.shape)
        self.data = data
        # print(dataloader.keys())
        self.single_kernal = False
        self.V_thr = 50
        self.T = 256                 # pattern duration ms
        self.dt = 1                  # time tick 1ms for lookup table
        self.tau_m = 20              # tau_m = 20ms
        self.tau_s = self.tau_m / 4

        # Set kernal
        # 相当于一个脉冲函数铺满整个时间窗
        t_list = np.array(list(range(0, self.T + self.dt, self.dt)))
        self.k1 = np.exp(-1 * t_list / self.tau_m)        # kernal 1
        self.k2 = None
        if not self.single_kernal:
            self.k2 = np.exp(-1 * t_list / self.tau_s)    # kernal 2
        t_list = -1 * \
            (np.array(list(range(0, 5 * self.tau_m + self.dt, self.dt))))
        self.V0 = 1 / np.max(np.exp(t_list / self.tau_m) -
                             np.exp(t_list / self.tau_s))
        # print(self.V0)

    def eventdriven(self):
        nOutputs = self.weights.shape[1]
        # print(nOutputs)
        nNeuronPerOutput = self.weights.shape[2]
        # print(nNeuronPerOutput)
        # nImages = 1 # len(self.PtnCell)
        # correctRate = np.zeros((1, self.maxEpoch))
        # dw_Past = np.zeros(self.weights.shape)
        output = [0]*11
        # visit mat_struct object
        # another method: data['PtnCellTst'][0].__dict__['AllVec']
        ptn = self.data.AllVec
        addr = self.data.AllAddr
        # label from 0-9
        # ptn = ptn[:, np.newaxis]
        # addr = addr[:, np.newaxis]
        # print('label:', label)
        # establish lookup table
        nAfferents = np.arange(
            1, len(ptn)+1, 1).reshape(len(ptn), 1)     # 1:len(ptn)+1
        P = np.hstack((ptn, addr, nAfferents))
        peak_delay = 0.462 * self.tau_m
        onlyP = np.unique(P[:, 0]).astype(float)    # int32 wrong
        onlyP = np.expand_dims(onlyP + peak_delay, axis=1)
        onlyP = np.hstack((onlyP, np.array(
            [[self.T]*len(onlyP)]).T)).min(1).reshape(len(onlyP), 1)    # array.T 转置
        fill_data = -1 * np.ones((len(onlyP), 1))
        Pd = np.hstack((onlyP, fill_data, fill_data))
        # print(P.shape)
        # print(Pd.shape)
        P = np.vstack((P, Pd))
        # 第一列升序排序索引 matlab默认的排序形式是 mergesort 巨坑
        idx = np.argsort(P[:, 0], kind='mergesort')
        P = P[idx, :]
        P = np.vstack((P, np.array([[self.T, -1, -1]])))    # add end
        numEvt = P.shape[0]
        for neuron in range(nOutputs):
            # print('neuron', neuron)
            for indNeuronPerOutput in range(nNeuronPerOutput):
                # print('indNeuronPerOutput', indNeuronPerOutput)
                out = False  # output
                Vmax = np.NINF  # 负无穷@@@@@@
                tmax = np.NINF
                fired = False
                t_last = -1
                t_latestRealEvt = np.NINF
                cnt_evts_of_same_timestamp = 1  # counter
                Vm = 0
                Vm_K1 = 0
                Vm_K2 = 0
                for i in range(numEvt):
                    t = P[i, 0]
                    addr_i = int(P[i, 1]) - 1
                    # print(addr_i)
                    c = P[i, 2]
                    delta_t = t - t_last
                    # print('t: %d, t_last: %d, delta_t: %d' %
                    #       (t, t_last, delta_t))
                    condition_lastVm_checkup = (
                        (delta_t > 0) and (i > 0)) or (i == numEvt - 1)
                    if not condition_lastVm_checkup:
                        if i != 0:
                            cnt_evts_of_same_timestamp = cnt_evts_of_same_timestamp + 1
                    else:
                        if Vm > Vmax:
                            Vmax = Vm
                            tmax = t_last
                        if Vm >= self.V_thr and fired is False:  # fire
                            fired = True
                            self.t_fire = t_last
                            out = True
                            if self.single_kernal:
                                Vm = 1.2 * Vm  # to make the output spike more noticeable
                        cnt_evts_of_same_timestamp = 1
                    if fired:
                        break
                    else:
                        lut_addr = int(round(delta_t / self.dt))
                        if lut_addr <= len(self.k1):
                            Sc1 = self.k1[lut_addr]
                        else:
                            Sc1 = 0
                        Vm_K1 = Sc1 * Vm_K1
                        if c != -1:
                            # print(self.weights[addr_i, neuron, indNeuronPerOutput])
                            Vm_K1 = Vm_K1 + self.V0 * \
                                self.weights[addr_i, neuron,
                                             indNeuronPerOutput]
                        if not self.single_kernal:
                            if lut_addr <= len(self.k2):
                                Sc2 = self.k2[lut_addr]
                            else:
                                Sc2 = 0
                            Vm_K2 = Sc2 * Vm_K2
                            if c != -1:
                                Vm_K2 = Vm_K2 + self.V0 * \
                                    self.weights[addr_i, neuron,
                                                 indNeuronPerOutput]
                            Vm = Vm_K1 - Vm_K2
                        else:
                            Vm = Vm_K1
                        if c != -1:
                            t_latestRealEvt = t
                        t_last = t
                if Vmax <= 0:
                    tmax = t_latestRealEvt
                if out:
                    # print(neuron, indNeuronPerOutput)
                    output[neuron] = output[neuron] + 1
        return output.index(max(output))
