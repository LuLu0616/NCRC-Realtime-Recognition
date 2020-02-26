function [weights, correctRate, CorrectFireRate, CorrectNonFireRate,epoch]= EventDrivenTempotron(weights, IsTraining, SimWhichSet, datafolder, MAXEPO, lmd)

if ~exist('datafolder','var')
    datafolder = pwd;
end

if ~exist('MAXEPO','var'), MAXEPO = 1000; end
if (IsTraining)
    maxEpoch = MAXEPO;
    load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn')
    PtnCell = PtnCellTrn;
    clear PtnCellTrn;
else
    maxEpoch = 1;   % if not training, only run 1 epoch
    if isequal(SimWhichSet, 'training set')
        load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn')
        PtnCell = PtnCellTrn;
        clear PtnCellTrn;
    elseif isequal(SimWhichSet,'testing set')
        load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTst')
        PtnCell = PtnCellTst;
        clear PtnCellTst;
    elseif isequal(SimWhichSet,'RealTime sample')
                if(rand>0.5)
                    load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTrn');
                    PtnCell{1} = PtnCellTrn{randperm(length(PtnCellTrn),1)};
                    clear PtnCellTrn;
                else
                    load([datafolder, '/', 'PtnCell_spk'], 'PtnCellTst');
                    PtnCell{1} = PtnCellTst{randperm(length(PtnCellTst),1)};
                    clear PtnCellTst;
                end
                % load TD;
                % PtnCell{1}=TD;
%                 TD = PtnCell{1};
%                 save('TD','TD')
    else
        error('Error: SimWhichSet can only be ''training set'' or ''testing set''!');
    end
end

if ~exist('lmd','var')
    lmd = 2e-2;
end
if ~exist('targetRate','var')
    targetRate = 1;
end

use_single_exponential = 1 ;

V_thr  = 50;

T = 256;        % pattern duration ms
dt = 1;             % time tick 1ms for lookup table
tau_m = 20;         % tau_m = 20 ms;
tau_s = tau_m/4;
mu = 0;

tau1 = tau_m;
tau2 = tau_s;

% lookup table
% 相当于一个脉冲函数铺满整个时间窗
t = 0:dt:T;
lut1 = exp(-t/tau1);
if ~use_single_exponential
    lut2 = exp(-t/tau2);
end

V0 = 1/max(exp(-(0:dt:5*tau1)/tau1)-exp(-(0:dt:5*tau1)/tau2));
nOutputs = size(weights, 2);
nNeuronPerOutput = size(weights, 3);
nImages = length(PtnCell);
correctRate = zeros(1, maxEpoch);
dw_Past = zeros(size(weights));

for epoch = 1:maxEpoch
    numTotSlices = 0;
    numCorrectSlices = 0;
    numTotFireSlices = 0;
    numCorrectFireSlices = 0;
    numTotNonFireSlices = 0;
    numCorrectNonFireSlices = 0;
    for iImage = randperm(nImages)
        
        
        PtnCell{iImage}.Tgt = false(nNeuronPerOutput * nOutputs, 1);
        PtnCell{iImage}.Out = false(nNeuronPerOutput * nOutputs, 1);
        
        addr = PtnCell{iImage}.AllAddr;     % spike address
        ptn  = PtnCell{iImage}.AllVec;      % spike timings
        lbl  = PtnCell{iImage}.Time_Chnl_Lbl;
        tgt = false(nNeuronPerOutput*nOutputs, 1);    % target
        tgt(lbl*nNeuronPerOutput+1: (lbl+1)*nNeuronPerOutput) = true;
        PtnCell{iImage}.Tgt = tgt;
        
        nAfferents = length(ptn);	% number of effective afferent. (e.g. 100)
        P = [ptn, addr, (1:nAfferents)'];
        
%         peak_delay = 0.462*tau1;
%         onlyP =  unique(P(:,1));
%         Pd = [min(onlyP+peak_delay, T), -1*ones(size(onlyP)), -1*ones(size(onlyP))]; % peak timings.
%         P = [P;Pd];
%         [~, idx_tmp] = sort(P(:,1), 'ascend');
%         P = P(idx_tmp, : ,:);
%         P = [P; T,-1,-1];
        
        numEvt = size(P, 1);
        for neuron = 1:nOutputs
            for indNeuronPerOutput = 1:nNeuronPerOutput
                out = false;        % output
                
                Vmax = -inf;
                tmax = -inf;
                t_fire = -inf;
                fired = false;
                t_last = -1;
                t_latestRealEvt = -inf;
                cnt_evts_of_same_timestamp = 1;  % counter
                Vm = 0;
                Vm_K1 = 0;
                Vm_K2 = 0;
                
                for i = 1:numEvt
                    t = P(i, 1);
                    addr_i = P(i, 2);
                    c = P(i, 3);
                    delta_t = t - t_last;
                    
                    condition_lastVm_checkup = ((delta_t > 0)&&(i > 1)) || (i==numEvt);
                    if ~condition_lastVm_checkup
                        if i ~= 1
                            cnt_evts_of_same_timestamp = cnt_evts_of_same_timestamp +1;
                        end
                    else
                        
                        if Vm > Vmax
                            Vmax = Vm;
                            tmax = t_last;
                        end
                        
                        if Vm >= V_thr && fired == false % fire
                            fired = true;
                            t_fire = t_last;
                            out = true;
                            if use_single_exponential
                                Vm = 1.2*Vm; % to make the output spike more noticeable
                            end
                        end
                        cnt_evts_of_same_timestamp = 1;     % important to reset this counter
                    end
                    
                    refractory = fired ;
                    if refractory
                        break;
                    else
                        lut_addr = round(delta_t/dt)+1;
                        if lut_addr <= length(lut1)
                            Sc1 = lut1(lut_addr);
                        else
                            Sc1 = 0;
                        end
                        Vm_K1 = Sc1 * Vm_K1;
                        if (c ~= -1)    % only for the original input spikes, not the dummy ones.
                            Vm_K1 = Vm_K1 + V0*weights(addr_i,neuron,indNeuronPerOutput);
                        end
                        if ~use_single_exponential
                            if lut_addr <= length(lut2)
                                Sc2 = lut2(lut_addr);
                            else
                                Sc2 = 0;
                            end
                            Vm_K2 = Sc2*Vm_K2;
                            if (c ~= -1)
                                Vm_K2 = Vm_K2 + V0*weights(addr_i,neuron,indNeuronPerOutput);
                            end
                            Vm = Vm_K1 - Vm_K2;
                        else
                            Vm = Vm_K1;
                        end
                        if c ~= -1
                            t_latestRealEvt = t;
                        end
                        t_last = t;
                    end
                end % end of events
                if Vmax <= 0
                    tmax = t_latestRealEvt;
                end
                indTotNeuronOutput = ((neuron-1)*nNeuronPerOutput+indNeuronPerOutput);
                PtnCell{iImage}.Out(indTotNeuronOutput) = out;
                %正确的个数,和总共计算的个数
                numCorrectSlices = numCorrectSlices + double((tgt(indTotNeuronOutput)==out)) ;
                numTotSlices = numTotSlices + 1;
                %正确的fire个数,和总共fire的个数
                numCorrectFireSlices = numCorrectFireSlices +  double((tgt(indTotNeuronOutput)==1)&(tgt(indTotNeuronOutput)==out));
                numTotFireSlices = numTotFireSlices + double((tgt(indTotNeuronOutput)==1));
                %没有的fire个数,和总没有fire的个数
                numCorrectNonFireSlices = numCorrectNonFireSlices + double((tgt(indTotNeuronOutput)==0)&(tgt(indTotNeuronOutput)==out));
                numTotNonFireSlices = numTotNonFireSlices + double((tgt(indTotNeuronOutput)==0));
                
                % ------------------ TRAINING start ------------------
                if (IsTraining)
                    if out ~= tgt(indTotNeuronOutput)   % error (update weights)
                        index = (P(:, 1) <= tmax) & (P(:, 2) ~= -1);
                        %找出最大电压前的所有事件
                        P1 = P(index, :);
                        % may have repeated addresses. i.e. multispike per afferent.
                        K_tmax  = zeros(nAfferents, 1);
                        K_tmax1 = zeros(nAfferents, 1);
                        K_tmax2 = zeros(nAfferents, 1);
                        ts = P1(:, 1);
                        addrs = P1(:, 2);
                        cs = P1(:, 3);
                        delta_ts = tmax - ts;
                        lut_addrs = round(delta_ts /dt) + 1;
                        Sc1s = zeros(size(lut_addrs));
                        
                        Sc1s(lut_addrs<=length(lut1)) = lut1(lut_addrs);
                        K_tmax1(cs) = V0 * Sc1s;
                        if ~use_single_exponential
                            Sc2s = zeros(size(lut_addrs));
                            Sc2s(lut_addrs<=length(lut2)) = lut2(lut_addrs);
                            K_tmax2(cs) = V0 * Sc2s;
                            K_tmax = K_tmax1 - K_tmax2;
                        else
                            K_tmax = K_tmax1;
                        end
                        
                        %合并相同地址的膜电压
                        addr_single = addrs * 0;
                        K_tmax_single = K_tmax * 0;
                        addrNum = 0;
                        for i = 1:length(addrs)
                            if addrs(i) ~= 0
                                addrNum = addrNum + 1;
                                index = (addrs == addrs(i));
                                k = K_tmax(index);
                                K_tmax_single(addrNum) = sum(k);
                                addr_single(addrNum) = addrs(i);
                                addrs(index) = 0;
                            end
                        end
                        
                        K_tmax_single = K_tmax_single(1:addrNum);
                        addr_single = addr_single(1:addrNum);
                        
                        if fired == false    % LTP
                            Dw = lmd * K_tmax_single;
                        else                 % LTD
                            Dw = -1 * lmd * K_tmax_single;
                        end
                        A1 = weights(addr_single, neuron, indNeuronPerOutput);	% weights(addr, neuron, indNeuronPerOutput);
                        dwPst = dw_Past(addr_single, neuron, indNeuronPerOutput);
                        A1 = A1 + Dw + mu*dwPst.*(Dw~=0);
                        weights(addr_single, neuron, indNeuronPerOutput) = A1; 	% ****
                        dwPst(Dw~=0) = Dw(Dw~=0);
                        dw_Past(addr_single, neuron, indNeuronPerOutput) = dwPst;
                    end
                end
                % ------------------ TRAINING end ------------------
            end % end of one NeuronPerOutput
        end % end of one neuron (population)
    end % end of one "Image"
    
    TrainedWt = weights;
    correctRate(epoch)= numCorrectSlices / numTotSlices;
    CorrectFireRate = numCorrectFireSlices/numTotFireSlices;
    CorrectNonFireRate = numCorrectNonFireSlices/numTotNonFireSlices;
    if(IsTraining), save([datafolder,'/','TrainedWt'],'TrainedWt','correctRate','epoch'); end
    if(IsTraining), timedLog( ['epoch: ', num2str(epoch), ', correct Fire rate: ', num2str(CorrectFireRate),...
            ', correct NonFire rate: ', num2str(CorrectNonFireRate), ', total correct rate: ', num2str(correctRate(epoch))] );
    end
    
    if(IsTraining)
        Rates = correctRate(max(1,epoch-9):epoch);  % 最近10次的最大正确率
        avgRate(epoch) = mean(Rates);               % 最近10次的平均正确率
        [~, idx1] = sort( -1* avgRate(max(1,epoch-9):epoch) );
        condition1 = correctRate(epoch)==1;         % all correct  本次训练全对
        condition2 = sum(Rates==Rates(1)) ==10;     % no change of correctRate for 10 consecutive epochs  10次连续的训练正确率都一样
        condition3 = isequal(idx1,1:10);            % avgRate decreases for 10 consecutive epochs         10次连续的训练正确率下降
        condition4 = (epoch>10) & ( sum(abs(Rates-mean(Rates))) < 1e-3 ); % rate almost saturates         训练10次以上并且几乎不变
        condition5 = correctRate(epoch)>= targetRate;                                                   % 正确率高于目标值
        if condition1 || condition5 || condition2 || condition3,
            % all correct || no change of correctRate for 10 consecutive epochs || avgRate decreases for 10 consecutive epochs
            if condition1, timedLog('Training ends: 100%% rate achieved'); end
            if condition2, timedLog('Training ends: no change of correcRate for 10 consecutive epochs'); end
            if condition3, timedLog('Training ends: avgRate decreases for 10 consecutive epochs'); end
            if condition5, timedLog(sprintf('target rate %.6f achieved',targetRate)); end
            correctRate = correctRate(1:epoch);
            break;     % break, no need to run more epochs.
        end
        if condition2 || condition3 || condition4
            % reduce the learning rate.
            if lmd> 1e-6
                lmd = lmd/2;
                timedLog( sprintf('lmd changed to %.6f',lmd) );
            else
                timedLog( 'corr rate almost saturates. stop training');
                correctRate = correctRate(1:epoch);
                break
            end
        end
        if epoch == maxEpoch
            timedLog(['Training ends: maxEpoch (' num2str(maxEpoch) ') achieved']);
        end
    end
end	% end of one epoch
if ~IsTraining
    if isequal(SimWhichSet, 'training set')
        PtnCellTrn = PtnCell;
        clear PtnCell;
        save([datafolder, '/', 'PtnCellTrn_out'], 'PtnCellTrn')
    elseif isequal(SimWhichSet, 'testing set')
        PtnCellTst = PtnCell;
        clear PtnCell;
        save([datafolder, '/', 'PtnCellTst_out'], 'PtnCellTst')
    elseif isequal(SimWhichSet, 'RealTime sample')
        PtnCellRealTime = PtnCell;
        clear PtnCell;
        save([datafolder, '/', 'PtnCellReal_out'], 'PtnCellRealTime')
    end
end
