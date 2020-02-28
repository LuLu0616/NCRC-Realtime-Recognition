import os
import random
import numpy


class Tempotron:
    def __init__(self, weights, IsTraining, PtnCell, maxEpoch, lmd):
        self.weights = weights
        self.IsTraining = IsTraining
        self.Ptncell = PtnCell
        self.maxEpoch = maxEpoch
        self.lmd = lmd

    def EventDrivenTempotron(self):
        '''if datafolder is None:
            datafolder = os.getcwd()
        if IsTraining:
            PtnCellTrn = None #todo: ****load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn')****
            PtnCell = PtnCellTrn
            del PtnCellTrn
        else:
            maxEpoch = 1
            if SimWhichSet == "training set":
                PtnCellTrn = None  # todo: ****load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn')****
                PtnCell = PtnCellTrn
                del PtnCellTrn
            elif SimWhichSet == "testing set":
                PtnCellTst = None  # todo: ****load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTst')****
                PtnCell = PtnCellTst
                del PtnCellTst
            elif SimWhichSet == "RealTime sample":  #****不要用, len(PenCell)有问题
                if random.random()>0.5:
                    PtnCellTrn = None  # todo: ****load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn')****
                    PtnCell = []
                    PtnCell.append(PtnCellTrn[:,0])
                    del PtnCellTrn
                else:
                    PtnCellTst = None  # todo: ****load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTst')****
                    PtnCell = []
                    PtnCell.append(PtnCellTst[:,0])
                    del PtnCellTst

            else:
                print("Error: SimWhichSet can only be ''training set'' or ''testing set''!")
                exit(1)'''

        targetRate = 1
        use_single_exponential = False
        V_thr = 50
        T = 256  # pattern duration ms
        dt = 1  # time tick 1ms for lookup table
        tau_m = 20  # tau_m = 20ms
        tau_s = tau_m / 4
        mu = 0
        tau1 = tau_m
        tau2 = tau_s
        # lookup table
        # 相当于一个脉冲函数铺满整个时间窗
        t = numpy.array(list(range(0, T + dt, dt)))
        lut1 = numpy.exp(-1 * t / tau1)
        lut2 = None
        if not use_single_exponential:
            lut2 = numpy.exp(-1 * t / tau2)
        V0 = 1 / numpy.max(numpy.exp(-1 * (numpy.array(list(range(0, 5 * tau1 + dt, dt)))) / tau1) - numpy.exp(
            -1 * (numpy.array(list(range(0, 5 * tau1 + dt, dt)))) / tau2))
        nOutputs = self.weights.shape[1]
        nNeuronPerOutput = self.weights.shape[2]
        nImages = 1 # len(self.PtnCell)
        correctRate = numpy.zeros((1, self.maxEpoch))
        dw_Past = numpy.zeros(self.weights.shape)

        for epoch in range(1, self.maxEpoch + 1):
            numTotSlices = 0
            numCorrectSlices = 0
            numTotFireSlices = 0
            numCorrectFireSlices = 0
            numTotNonFireSlices = 0
            numCorrectNonFireSlices = 0
            Tgt = {}  # ****本来是PtnCell{iImage}.Tgt，这里单开一个字典****
            Out = {}  # 同上

            for iImage in numpy.random.permutation(numpy.arange(nImages)):
                Tgt[iImage] = numpy.zeros((nNeuronPerOutput * nOutputs, 1), dtype=bool)
                Out[iImage] = numpy.zeros((nNeuronPerOutput * nOutputs, 1), dtype=bool)
                addr = self.Ptncell[iImage].AllAddr  # *****todo: addr = PtnCell{iImage}.AllAddr; ****   @@@要从0开始@@@
                ptn = self.Ptncell[iImage].AllVec  # *****todo: ptn  = PtnCell{iImage}.AllVec;* ***
                # lbl = None  # *****todo: lbl = PtnCell{iImage}.Time_Chnl_Lbl;*****
                # tgt = numpy.zeros((nNeuronPerOutput * nOutputs, 1), dtype=bool)
                # tgt[lbl * nNeuronPerOutput: (lbl + 1) * nNeuronPerOutput] = [True]
                # *****tgt(lbl*nNeuronPerOutput+1: (lbl+1)*nNeuronPerOutput) = true;*****
                # Tgt[iImage] = tgt
                nAfferents = len(ptn)  # *****nAfferents = length(ptn);*****
                # 有无简洁写法↓
                # print(ptn.shape)
                # print(addr.shape)
                P = numpy.hstack((ptn, addr))
                # print(P.shape)
                x = numpy.ones((nAfferents, 1))
                P = numpy.hstack((P, x))
                # print(P.shape)
                peak_delay = 0.462 * tau1
                onlyP = P[:, 0]
                # print(onlyP.shape)
                onlyP = numpy.unique(onlyP)
                # *****Pd = [min(onlyP + peak_delay, T), -1 * ones(size(onlyP)), -1 * ones(size(onlyP))]*****
                for i in range(len(onlyP)):
                    onlyP[i] = min(onlyP[i] + peak_delay, T)
                onlyP = numpy.expand_dims(onlyP, axis=1)
                # print(onlyP.shape)
                Pd = numpy.hstack((onlyP, -1 * numpy.ones((len(onlyP), 1))))
                Pd = numpy.hstack((Pd, -1 * numpy.ones((len(onlyP), 1))))
                # print(Pd.shape)
                P = numpy.vstack((P, Pd))
                # print(P.shape)
                # ******[~, idx_tmp] = sort(P(:, 1), 'ascend');*****
                # ******P = P(idx_tmp,:,:); ******
                # @@@@@@@对P按照第一个列的大小 对每个行进行升序排序@@@@@@@
                P = P[numpy.lexsort(P[:, ::-1].T)]
                P = numpy.vstack((P, numpy.array([T, -1, -1])))
                # print(P.shape)
                # 有无简洁写法↑
                numEvt = P.shape[0]
                for neuron in range(nOutputs):
                    for indNeuronPerOutput in range(nNeuronPerOutput):
                        out = False  # output
                        Vmax = numpy.NINF  # @@@@@@负无穷@@@@@@
                        tmax = numpy.NINF
                        t_fire = numpy.NINF
                        fired = False
                        t_last = -1
                        t_latestRealEvt = numpy.NINF
                        cnt_evts_of_same_timestamp = 1  # counter
                        Vm = 0
                        Vm_K1 = 0
                        Vm_K2 = 0
                        for i in range(numEvt):
                            t = P[i, 0]
                            addr_i = int(P[i, 1])
                            c = P[i, 2]
                            delta_t = t - t_last
                            condition_lastVm_checkup = ((delta_t > 0) and (i > 0)) or (i == numEvt - 1)
                            if not condition_lastVm_checkup:
                                if i != 1:
                                    cnt_evts_of_same_timestamp = cnt_evts_of_same_timestamp + 1
                            else:
                                if Vm > Vmax:
                                    Vmax = Vm
                                    tmax = t_last
                                if Vm >= V_thr and fired == False:  # fire
                                    fired = True
                                    t_fire = t_last
                                    out = True
                                    if use_single_exponential:
                                        Vm = 1.2 * Vm  # to make the output spike more noticeable
                                cnt_evts_of_same_timestamp = 1
                            refractory = fired
                            if refractory:
                                break
                            else:
                                lut_addr = int(round(delta_t / dt))  # @@@@@去掉了+1, python下标@@@@@
                                if lut_addr < len(lut1):
                                    Sc1 = lut1[lut_addr]
                                else:
                                    Sc1 = 0
                                Vm_K1 = Sc1 * Vm_K1
                                if c != -1:
                                    Vm_K1 = Vm_K1 + V0 * self.weights[addr_i, neuron, indNeuronPerOutput]
                                if not use_single_exponential:
                                    if lut_addr < len(lut2):  # @@@@@原来是<=@@@@@@
                                        Sc2 = lut2[lut_addr]
                                    else:
                                        Sc2 = 0
                                    Vm_K2 = Sc2 * Vm_K2
                                    if (c != -1):
                                        Vm_K2 = Vm_K2 + V0 * self.weights[addr_i, neuron, indNeuronPerOutput]
                                    Vm = Vm_K1 - Vm_K2
                                else:
                                    Vm = Vm_K1
                                if c != -1:
                                    t_latestRealEvt = t
                                t_last = t
                        if Vmax <= 0:
                            tmax = t_latestRealEvt
                        if out:
                            print(neuron, indNeuronPerOutput)
