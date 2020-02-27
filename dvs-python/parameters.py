import math


ispool1 = 1
ispool2 = 1
poolsize1 = 2
poolsize2 = 2
imsize0 = [29, 29]
imsize1 = [math.ceil(i / poolsize1) for i in imsize0]
imsize2 = [math.ceil(i / poolsize2) for i in imsize1]
maxEpoch = 10
noutputs = 10
lmd = 1e-1
nNeuronPerOutput = 10
nAfferents = imsize2[0] * imsize2[1] * 18

reRunAll = 1
reGetFeature = 1
reGenPtnCell = 1
reSplitDataset = 1
reSpkConvert = 1
reInitWeights = 1
reTrnWeights = 1
continueTrnWts = 0
reSimulation = 1
reAnalyzeResults = 1
reRealTimeTest = 1
