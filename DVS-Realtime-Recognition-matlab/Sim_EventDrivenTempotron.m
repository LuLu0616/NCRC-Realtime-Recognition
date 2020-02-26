function  Sim_EventDrivenTempotron(weights, TD)
PtnCell = TD;
use_single_exponential = 0 ;
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

PtnCell.Tgt = false(nNeuronPerOutput * nOutputs, 1);
PtnCell.Out = false(nNeuronPerOutput * nOutputs, 1);

addr = PtnCell.AllAddr;     % spike address
ptn  = PtnCell.AllVec;      % spike timings
lbl  = PtnCell.Time_Chnl_Lbl;
tgt = false(nNeuronPerOutput*nOutputs, 1);    % target
tgt(lbl*nNeuronPerOutput+1: (lbl+1)*nNeuronPerOutput) = true;
PtnCell.Tgt = tgt;

nAfferents = length(ptn);	% number of effective afferent. (e.g. 100)
P = [ptn, addr, (1:nAfferents)'];

peak_delay = 0.462*tau1;
onlyP =  unique(P(:,1));
Pd = [min(onlyP+peak_delay, T), -1*ones(size(onlyP)), -1*ones(size(onlyP))]; % peak timings.
P = [P;Pd];
[~, idx_tmp] = sort(P(:,1), 'ascend');
P = P(idx_tmp, : ,:);
P = [P; T,-1,-1];

numEvt = size(P, 1);

for neuron = 1:nOutputs
    for indNeuronPerOutput = 1:nNeuronPerOutput
        out = false;        % output
        Vmax = -inf;
        fired = false;
        t_last = -1;
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
                end
                if Vm >= V_thr && fired == false % fire
                    fired = true;
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
                t_last = t;
            end
        end % end of events
        indTotNeuronOutput = ((neuron-1)*nNeuronPerOutput+indNeuronPerOutput);
        PtnCell.Out(indTotNeuronOutput) = out;
    end % end of one NeuronPerOutput
end % end of one neuron (population)

Out = PtnCell.Out;
freqs = zeros(nOutputs,1);
for neuron = 1:nOutputs
    for indNeuronPerOutput = 1:nNeuronPerOutput
        indTotNeuronOutput = ((neuron-1)*nNeuronPerOutput+indNeuronPerOutput);
        out = Out(indTotNeuronOutput,1);
        if out==1
            freqs(neuron,1) = freqs(neuron,1) +1;
        end
    end
end
decision = find(freqs==max(freqs),1) - 1;
timedLog('RealTime Test:');
fprintf('\t actual lable:  %2d \n',  lbl);
fprintf('\t OutPut      :  %2d \n',  decision);
end


