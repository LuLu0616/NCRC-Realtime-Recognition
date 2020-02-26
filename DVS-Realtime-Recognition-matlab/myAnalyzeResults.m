function [ CorrRate_Trn, CorrRate_Tst, decision_Trn, decision_Tst, freqsCell_Trn, freqsCell_Tst ] = ...
    myAnalyzeResults( nOutput, nNeuronPerOutput, datafolder )
%AnalyzeResults_withRedundancy 

if ~exist('datafolder','var')
    datafolder = pwd;
end
load([datafolder, '/', 'PtnCellTrn.mat'])
load([datafolder, '/', 'PtnCellTrn_out.mat'])
[CorrRate_Trn, decision_Trn, freqsCell_Trn] = AnalyzeResults_withRedundancy_eachSliceToAllNeuron_Core(PtnCellTrn, nOutput, nNeuronPerOutput);
load([datafolder, '/', 'PtnCellTst.mat'])
load([datafolder, '/', 'PtnCellTst_out.mat'])
[CorrRate_Tst, decision_Tst, freqsCell_Tst] = AnalyzeResults_withRedundancy_eachSliceToAllNeuron_Core(PtnCellTst, nOutput, nNeuronPerOutput);

end

function [CorrRate, decision,freqsCell] = AnalyzeResults_withRedundancy_eachSliceToAllNeuron_Core(PtnCellTst, nOutput,nNeuronPerOutput)

N = length(PtnCellTst);
decision = cell(1,N);
freqsCell = cell(1,N);
numCorrect = 0;
numTotSlice = 0;

for i = 1:N
    
    lbl    = PtnCellTst{i}.Time_Chnl_Lbl(3,1);

    Out = PtnCellTst{i}.Out;
    nSlices = size(Out,2);
    freqs = zeros(nOutput,nSlices);
    
    for indSlice = 1:nSlices
        for neuron = 1:nOutput
            for indNeuronPerOutput = 1:nNeuronPerOutput

                indTotNeuronOutput = ((neuron-1)*nNeuronPerOutput+indNeuronPerOutput);
                out = Out(indTotNeuronOutput,indSlice);

                if out==1
                    freqs(neuron,indSlice) = freqs(neuron,indSlice) +1;
                end
            end
        end
    end
    
    freqsCell{i} = freqs;
    
%     freqs1 = sum(freqs,2);
%     decision(i) = find(freqs1==max(freqs1),1) - 1;
%     numCorrect = numCorrect + double(decision(i)==lbl(1));

    decision{i} = zeros(1,nSlices);
    for indSlice = 1:nSlices
        freqs1 = freqs(:,indSlice);
        %fire最多的就认为是那个类,一样的取第一个
        decision{i}(indSlice) = find(freqs1==max(freqs1),1) - 1;
        %和目标一样化就是正确的
        numCorrect = numCorrect + double(decision{i}(indSlice)==lbl(1));
        numTotSlice = numTotSlice + 1;
    end


end

% CorrRate = numCorrect/N;
CorrRate = numCorrect/numTotSlice;

end

