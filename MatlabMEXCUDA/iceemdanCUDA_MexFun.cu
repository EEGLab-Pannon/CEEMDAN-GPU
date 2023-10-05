/*
MIT License

Copyright (c) 2023 Electrical Brain Imaging Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cudaICEEMDAN.cuh"
#define PI 3.1415926

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    //===============define the variables for taking parameters================
    mxGPUArray const *mxInputData;
    mxGPUArray const *mxInputDataIndex;
    mxGPUArray const *mxOutputData; // just a dummy
    mxGPUArray *mxIMFs;

    int numNoise;
    int num_IMFs;
    int max_iter;
    float noiseStrength;

    float* h_y, * d_y;
    int* h_x, * d_x;
    float* d_IMFs;

    mxInitGPU();
    //===============convert variables for real using================
    mxInputData = mxGPUCreateFromMxArray(prhs[0]);
    mxInputDataIndex = mxGPUCreateFromMxArray(prhs[1]);
    mxOutputData = mxGPUCreateFromMxArray(prhs[2]);
    noiseStrength = *(float*)mxGetData(prhs[3]);
    numNoise = *(int*)mxGetData(prhs[4]);
    max_iter = *(int*)mxGetData(prhs[5]);
    num_IMFs = *(int*)mxGetData(prhs[6]);

    d_y = (float*)(mxGPUGetDataReadOnly(mxInputData));
    d_x = (int*)(mxGPUGetDataReadOnly(mxInputDataIndex));
    mxIMFs = mxGPUCreateGPUArray(mxGPUGetNumberOfDimensions(mxOutputData),
                                 mxGPUGetDimensions(mxOutputData),
                                 mxGPUGetClassID(mxOutputData),
                                 mxGPUGetComplexity(mxOutputData),
                                 MX_GPU_DO_NOT_INITIALIZE);
    d_IMFs = (float*)(mxGPUGetData(mxIMFs));
    int SignalLength = (int)(mxGPUGetNumberOfElements(mxInputData));

    //ceemdan processing
    double exeTime = iceemdanS(numNoise, SignalLength, num_IMFs, max_iter, d_x, d_y, d_IMFs, noiseStrength);
    plhs[0] = mxGPUCreateMxArrayOnGPU(mxIMFs);
}
