%% compile the source code to your local environment
setenv("MW_NVCC_PATH","C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\bin")
mexcuda 'NVCCFLAGS=-gencode=arch=compute_80,code=sm_80' '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64' -lcublas -lcusparse -lcurand iceemdanCUDA_MexFun.cu cudaICEEMDAN.cu statistics.cu
%% sample for two-tone synthetic signal decomposition
dataIdx = [0:999];
s1 = zeros(1,1000);
s1([501:750]) = sin(2*pi*0.255*[0:249]);
s2 = sin(2*pi*0.065*dataIdx);
inputSignal = s1 + s2;

mxInputData = gpuArray(single(inputSignal)); % input1
mxInputDataIndex = gpuArray(int32(dataIdx)); % input2
noiseStrength = single(0.2); % input4
numNoise = int32(100); % input5
max_iter = int32(100); % input6
num_IMFs = int32(2); % input7
mxOutputData = gpuArray(single(zeros(int32(length(inputSignal)), num_IMFs))); % input3

IMFs = iceemdanCUDA_MexFun(mxInputData, mxInputDataIndex, mxOutputData, noiseStrength, numNoise, max_iter, num_IMFs);
modes = gather(IMFs);
%% sample for real EEG signal decomposition
load('eegSampleDataCH4.mat')
inputSignal = eegSampleDataCH4;
dataIdx = [0:length(inputSignal) - 1];

mxInputData = gpuArray(single(inputSignal)); % input1
mxInputDataIndex = gpuArray(int32(dataIdx)); % input2
noiseStrength = single(0.2); % input4
numNoise = int32(200); % input5
max_iter = int32(300); % input6
num_IMFs = int32(12); % input7
mxOutputData = gpuArray(single(zeros(int32(length(inputSignal)), num_IMFs))); % input3

IMFs = iceemdanCUDA_MexFun(mxInputData, mxInputDataIndex, mxOutputData, noiseStrength, numNoise, max_iter, num_IMFs);
modes = gather(IMFs);
