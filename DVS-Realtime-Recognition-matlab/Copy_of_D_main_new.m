clear all;clc;
%% Feature parameter
ispool1 = 1;
ispool2 = 1;
poolsize1 = 2;
poolsize2 = 2;
imsize0 = [29 29];  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%需要改1（nminist34*34 gesture128*128）
imsize1 = ceil(imsize0/poolsize1);
imsize2 = ceil(imsize1/poolsize2);
maxEpoch = 5;
isdenoise = 0;

%% parameter
nOutputs = 10;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%需要改2（输出的类别）
%%
data_directory = 'D:/study/holiday2020/AER_DATA/Final_DVS_Cat/ExperienceData/MNIST_DVS/MNIST_DVS_full';%%%%%%%%%%%%%%%%需要改3（数据集）
working_directory = 'D:/study/holiday2020/AER_DATA/Final_DVS_Cat/MY_OutPut/HOTS_TEMPTRON_MNISTDVSfull_new';%%%%%%%%%%%%%%%%%%%%需要改4（生成的结果）

working_Feature = [working_directory,'/Feature'];
if ~exist(working_Feature, 'dir')
    mkdir(working_Feature);
end
Working_OUT = [working_directory, '/Output'];
if ~exist(Working_OUT , 'dir')
    mkdir(Working_OUT);
end
%%
reRunAll         = 0;
reGetFeature     = 0;
reGenPtnCell     = 0;
reSplitDataset   = 0;
reSpkConvert     = 0;
reInitWeights    = 0;
reTrnWeights     = 0;
continueTrnWts   = 1;
reSimulation     = 1;
reAnalyzeResults = 1 ;
reRealTimeTest = 1;
%% STS参数
global params;
params.layer1.radius = 2;
params.layer1.tau = 20e3;
params.layer1.num_feature = 6;
params.layer1.C = zeros(5, 5, 6);
params.layer1.count = 0;
params.layer1.alpha = 0;
params.layer1.beta = 0;
params.layer1.pk = ones(6, 1);
params.layer1.image_size = imsize0;

params.layer2.radius = 4;
params.layer2.tau = 200e3;
params.layer2.num_feature = 18;
params.layer2.C = zeros(9, 9, 18);
params.layer2.count = 0;
params.layer2.alpha = 0;
params.layer2.beta = 0;
params.layer2.pk = ones(18, 1);
params.layer2.image_size = imsize1;

if reRunAll || reGetFeature
    Hsize = 0;
    tic
    fprintf('\nRandomly Extracting Feature spikes for All Data \n');
    classes = dir(data_directory);
    classes(1:2) = [];
    matrix = [];
    
    for class_i = randperm(length(classes))
        files = dir([data_directory, '/', classes(class_i).name]);
        files(1:2) = [];
        
        if ~exist(([working_Feature, '/',classes(class_i).name]), 'dir')
            mkdir([working_Feature, '/',classes(class_i).name]);
        end
        
        for file_i = randperm(length(files))
            matrix = [matrix; class_i file_i];
        end
    end
    matrix = matrix(randperm(length(matrix)),:);
    for j = 1:2
        for i = randperm(length(matrix))
            class_i = matrix(i,1);
            files = dir([data_directory, '/', classes(class_i).name]);
            files(1:2) = [];
            file_i = matrix(i,2);
            load([data_directory, '/', classes(class_i).name, '/', files(file_i).name]);
            TD = D_pre_process(TD,isdenoise);
            
            TD  = D_timesurface(TD,1,ispool1,poolsize1,5e3); %input,hotlayer,ispool,pooling_extent,refractory_period
            TD  = D_timesurface(TD,2,ispool2,poolsize2,5e3);
            % save([working_Feature, '/',classes(class_i).name, '/', files(file_i).name(1:end-4)], 'TD');
        end
    end
    %% 
 
        for i = randperm(length(matrix))
            class_i = matrix(i,1);
            files = dir([data_directory, '/', classes(class_i).name]);
            files(1:2) = [];
            file_i = matrix(i,2);
            load([data_directory, '/', classes(class_i).name, '/', files(file_i).name]);
            TD = D_pre_process(TD,isdenoise);
            
            TD  = Sim_timesurface(TD,1,ispool1,poolsize1,5e3); %input,hotlayer,ispool,pooling_extent,refractory_period
            TD  = Sim_timesurface(TD,2,ispool2,poolsize2,5e3);
            save([working_Feature, '/',classes(class_i).name, '/', files(file_i).name(1:end-4)], 'TD');
        end

    fprintf('\nFeature Extraction End ...\n');
    save([working_directory,'/','STSF_params'], 'params');
end

%%
file = [Working_OUT, '/', 'PtnCell.mat'];
if reRunAll || reGenPtnCell || ~exist(file,'file')
    PtnCell = Feature2PtnCell(working_Feature,imsize2);
    save([Working_OUT,'/PtnCell.mat'],'PtnCell');
end
%%
file = [Working_OUT, '/', 'PtnCell_raw.mat'];
if reRunAll || reSplitDataset || ~exist(file,'file')
    ratio_Trn = 6/7;
    load([Working_OUT, '/', 'PtnCell.mat'])
    [PtnCellTrn, PtnCellTst, indTrn, indTst,maxT] = RandSplit(PtnCell, ratio_Trn, nOutputs); %#ok<*ASGLU>
    save(file, 'PtnCellTrn', 'PtnCellTst', 'indTrn', 'indTst','maxT')
end
%%
file = [Working_OUT, '/', 'PtnCell_spk.mat'];
if reRunAll || reSpkConvert|| ~exist(file,'file')
    MyNormSpkPtn(Working_OUT);
end

clear PtnCell
clear PtnCellTrn PtnCellTst indTrn indTst

%% --- learning----

lmd = 1e-1;
nNeuronPerOutput = 10;
numFilters = 18;
nAfferents = imsize2(1)*imsize2(2)*numFilters;

mu_init_wt = 0;
sigma_init_wt = 0.1;
file = [Working_OUT,'/','weights0.mat'];
if ( (reRunAll) || (reInitWeights) || ~exist(file,'file') )
    timedLog('Initiating weights start...');
    weights = mu_init_wt + sigma_init_wt * rand(nAfferents, nOutputs, nNeuronPerOutput);
    save (file, 'weights')
    % else
    %     load (file)
end

%% training
tic
IsTraining = 1;
SimWhichSet = 'training set';
file = [Working_OUT, '/', 'TrainedWt.mat'];
TrnWtsFromInit = reRunAll || reTrnWeights || ~exist(file,'file');
if TrnWtsFromInit || continueTrnWts
    if (TrnWtsFromInit)
        timedLog('Start Training ...');
        [TrainedWt, correctRate, ~, ~, ~] = EventDrivenTempotron(weights, IsTraining, SimWhichSet,Working_OUT, maxEpoch, lmd);
    elseif (continueTrnWts)
        timedLog('Continue Training ..');
        load(file)
        [TrainedWt, correctRate, ~, ~, ~] = EventDrivenTempotron(TrainedWt, IsTraining, SimWhichSet, Working_OUT, maxEpoch, lmd);
    end
    timeTrain = toc / 60;    % minutes
    if timeTrain < 60
        timedLog(['Training finished, time taken: ', num2str(timeTrain), ' minutes'])
    else
        timedLog(['Training finished, time taken: ', num2str(timeTrain/60), ' hours'])
    end
    save(file, 'TrainedWt', 'correctRate');
end
load(file)

%% Simulation
tic
IsTraining = 0;
file = [Working_OUT, '/', 'RawResults.mat'];
if reRunAll || reSimulation || ~exist(file, 'file') || ...
        ~exist([Working_OUT, '/', 'PtnCellTrn_out.mat'], 'file') || ...
        ~exist([Working_OUT, '/', 'PtnCellTst_out.mat'], 'file')
    timedLog('Start Simulation ..');
    [TrainedWt, RR_trn, RR_trn_TP, RR_trn_TN,~] = EventDrivenTempotron(TrainedWt, IsTraining, 'training set', Working_OUT, 1, lmd);
    [TrainedWt, RR_tst, RR_tst_TP, RR_tst_TN,~] = EventDrivenTempotron(TrainedWt, IsTraining, 'testing set', Working_OUT, 1, lmd);
    timeSim = toc / 60;   % minutes
    if timeSim < 60
        timedLog(['Simulation finished, time taken: ',num2str(timeSim),' minutes'])
    else
        timedLog(['Simulation finished, time taken: ',num2str(timeSim/60), ' hours'])
    end
    save(file, 'RR_trn', 'RR_trn_TP', 'RR_trn_TN', 'RR_tst', 'RR_tst_TP', 'RR_tst_TN');
end
load(file)
fprintf('=============================================================\n');
fprintf('Raw correct Rate (i.e. rate of correctly responded neurons for all slices):\n');
fprintf('\t training set: total succ rate %.2f %%, \t true positive %.2f %%, \t true negative %.2f %% \n',  RR_trn*100, RR_trn_TP*100, RR_trn_TN*100);
fprintf('\t testing set : total succ rate %.2f %%, \t true positive %.2f %%, \t true negative %.2f %% \n',  RR_tst*100, RR_tst_TP*100, RR_tst_TN*100);


% Analyze Results
file = [Working_OUT, '/', 'FinalResults.mat'];
if reRunAll || reAnalyzeResults || ~exist(file,'file')
    [ FinalCorrRate_Trn, FinalCorrRate_Tst, ~, ~, ~, ~ ] = ...
        AnalyzeResults_withRedundancy_eachSliceToAllNeuron(nOutputs, nNeuronPerOutput, Working_OUT);
    save (file, 'FinalCorrRate_Trn', 'FinalCorrRate_Tst')
end
load (file)
fprintf('=============================================================\n');
fprintf('Final correct Rate (i.e. succ rate of categorization, by majority voting):\n');
fprintf('\t training set:  %.2f %% \n',  FinalCorrRate_Trn*100);
fprintf('\t testing set :  %.2f %% \n',  FinalCorrRate_Tst*100);
%% realtime test
if reRunAll || reRealTimeTest
    file = [Working_OUT, '/', 'TrainedWt.mat'];
    IsTraining = 0;
    load(file)
    timedLog('Start RealTime Simulation ..');
    for ii = 1:20
        [~, ~, ~, ~,~] = EventDrivenTempotron(TrainedWt, IsTraining, 'RealTime sample', Working_OUT, 1, lmd);
        [ lbl, Output] = AnalyzeResults_RealTime(nOutputs, nNeuronPerOutput,Working_OUT );
        
        fprintf('=============================================================\n');
        timedLog('RealTime Test:');
        fprintf('\t actual lable:  %2d \n',  lbl);
        fprintf('\t OutPut      :  %2d \n',  Output);
    end
end