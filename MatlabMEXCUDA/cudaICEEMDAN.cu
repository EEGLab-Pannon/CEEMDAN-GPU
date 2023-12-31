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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <tuple>
#include <omp.h>
#include <dirent.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cub/cub.cuh> 
#include <cuda.h>
#include "statistics.cuh"

template <typename real_t>
__global__ void produceFirstIMF(real_t* d_IMFs, real_t* d_running, real_t* d_noisedSignal, real_t* d_currentModes, real_t* d_forNext, size_t numNoise, size_t signalLength)
{
    int samplesIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (samplesIdx < signalLength)
    {
        real_t tem = 0;
        for (int i = 0; i < numNoise; i++)
        {
            tem = tem + (1.0 / numNoise) * (d_noisedSignal[i * signalLength + samplesIdx] - d_currentModes[i * signalLength + samplesIdx]);
        }
        d_forNext[samplesIdx] = tem; // d_forNext is the medias/aux
        d_IMFs[samplesIdx] = d_running[samplesIdx] - tem;
    }
}

template <typename real_t>
__global__ void standardizeRunning(real_t* d_running, size_t SignalLength, real_t* d_singleChannelVariance)
{
    int samplesIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (samplesIdx < SignalLength)
    {
        d_running[samplesIdx] = d_running[samplesIdx] * (1.0 / sqrt(d_singleChannelVariance[0]));
    }
}

template <typename real_t>
__global__ void addNoise(real_t* d_noisedSignal, real_t* d_running, real_t* d_whiteNoiseModes, size_t SignalLength, real_t noiseStrength, real_t* d_channelVariance)
{
    int samplesIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;

    if (samplesIdx < SignalLength)
    {
        d_noisedSignal[noiseIdx * SignalLength + samplesIdx] = d_running[samplesIdx] + noiseStrength * d_whiteNoiseModes[noiseIdx * SignalLength + samplesIdx] * (1.0 / sqrt(d_channelVariance[noiseIdx]));
    }
}

template <typename real_t>
__global__ void addNoise2(real_t* d_noisedSignal, real_t* d_running, real_t* d_whiteNoiseModes, size_t SignalLength, real_t noiseStrength, real_t* d_singleChannelVariance)
{
    int samplesIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;

    if (samplesIdx < SignalLength)
    {
        d_noisedSignal[noiseIdx * SignalLength + samplesIdx] = d_running[samplesIdx] + noiseStrength * d_whiteNoiseModes[noiseIdx * SignalLength + samplesIdx] * sqrt(d_singleChannelVariance[0]);
    }
}

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_max(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMaxFlag, size_t SignalLength) {

    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        real_t value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        //coord_t coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        real_t up = __shfl_up_sync(0xffffffff, value, 1);
        real_t down = __shfl_down_sync(0xffffffff, value, 1);
        if (value > up && value > down)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }

        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMaxFlag[blockIdx.y * SignalLength] = 1;
            d_sparseMaxFlag[blockIdx.y * SignalLength + SignalLength - 1] = 1;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_min(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMinFlag, size_t SignalLength) {

    int channelElementsIndex = blockIdx.x * blockDim.x + threadIdx.x;
    //int allElementsIndex = blockIdx.y * SignalLength + blockIdx.x * blockDim.x + threadIdx.x;

    int warpFlag = channelElementsIndex / 32;

    if ((channelElementsIndex - 2 * warpFlag) < SignalLength)
    {
        real_t value = d_ProjectSignals[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];
        //coord_t coord = d_multiChannelIndex[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)];

        real_t up = __shfl_up_sync(0xffffffff, value, 1);
        real_t down = __shfl_down_sync(0xffffffff, value, 1);

        if (value < up && value < down)
        {
            d_sparseMinFlag[blockIdx.y * SignalLength + (channelElementsIndex - 2 * warpFlag)] = 1;
        }
        // for temporary, set edge points as it were 
        if (channelElementsIndex == 0)
        {
            d_sparseMinFlag[blockIdx.y * SignalLength] = 1;
            d_sparseMinFlag[blockIdx.y * SignalLength + SignalLength - 1] = 1;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void select_extrema_max(coord_t* d_sparseMaxFlag, real_t* d_noisedSignal, coord_t* d_noisedSignalIndex, coord_t* d_MaxScanResult, /*coord_t* d_ScanResultZeroCross,*/
    real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, size_t SignalLength, coord_t* d_num_extrema_max /*coord_t* d_num_zeroCrossPoints*/)
{
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointsIdx < SignalLength)
    {
        real_t currentValue = d_noisedSignal[noiseIdx * SignalLength + pointsIdx];
        coord_t currentIndex = d_noisedSignalIndex[noiseIdx * SignalLength + pointsIdx];
        coord_t currentFlag = d_sparseMaxFlag[noiseIdx * SignalLength + pointsIdx];
        coord_t storeLocation = d_MaxScanResult[noiseIdx * SignalLength + pointsIdx];

        if (currentFlag != 0)
        {
            d_compactMaxValue[noiseIdx * SignalLength + storeLocation] = currentValue;
            d_compactMaxIndex[noiseIdx * SignalLength + storeLocation] = currentIndex;
        }
        if (pointsIdx == SignalLength - 1)
        {
            d_num_extrema_max[noiseIdx] = storeLocation + 1;
            //d_num_zeroCrossPoints[noiseIdx] = d_ScanResultZeroCross[noiseIdx * SignalLength + pointsIdx] + 1;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void select_extrema_min(coord_t* d_sparseMinFlag, real_t* d_noisedSignal, coord_t* d_noisedSignalIndex,
    coord_t* d_MinScanResult, real_t* d_compactMinValue, coord_t* d_compactMinIndex, size_t SignalLength,
    coord_t* d_num_extrema_min)
{
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointsIdx < SignalLength)
    {
        real_t currentValue = d_noisedSignal[noiseIdx * SignalLength + pointsIdx];
        coord_t currentIndex = d_noisedSignalIndex[noiseIdx * SignalLength + pointsIdx];
        coord_t currentFlag = d_sparseMinFlag[noiseIdx * SignalLength + pointsIdx];
        coord_t storeLocation = d_MinScanResult[noiseIdx * SignalLength + pointsIdx];

        if (currentFlag != 0)
        {
            d_compactMinValue[noiseIdx * SignalLength + storeLocation] = currentValue;
            d_compactMinIndex[noiseIdx * SignalLength + storeLocation] = currentIndex;
        }

        if (pointsIdx == SignalLength - 1)
        {
            d_num_extrema_min[noiseIdx] = storeLocation + 1;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void setBoundaryMax(real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, coord_t* d_MaxScanResult, size_t SignalLength)
{
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.x;
    int pointsIdx = threadIdx.x;

    if ((pointsIdx == 0) && (d_MaxScanResult[noiseIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_max, t_max;

        coord_t storeLocation_max = d_MaxScanResult[noiseIdx * SignalLength + pointsIdx];
        coord_t loc_max = noiseIdx * SignalLength + storeLocation_max;

        slope_max = (d_compactMaxValue[loc_max + 2] - d_compactMaxValue[loc_max + 1]) / (d_compactMaxIndex[loc_max + 2] - d_compactMaxIndex[loc_max + 1]);
        t_max = d_compactMaxValue[loc_max + 1] - slope_max * (d_compactMaxIndex[loc_max + 1] - d_compactMaxIndex[loc_max]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
    }

    if ((pointsIdx == 1) && (d_MaxScanResult[noiseIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_max, t_max;

        coord_t storeLocation_max = d_MaxScanResult[noiseIdx * SignalLength + SignalLength - 1];
        coord_t loc_max = noiseIdx * SignalLength + storeLocation_max;

        slope_max = (d_compactMaxValue[loc_max - 1] - d_compactMaxValue[loc_max - 2]) / (d_compactMaxIndex[loc_max - 1] - d_compactMaxIndex[loc_max - 2]);
        t_max = d_compactMaxValue[loc_max - 1] + slope_max * (d_compactMaxIndex[loc_max] - d_compactMaxIndex[loc_max - 1]);

        if (t_max > d_compactMaxValue[loc_max])
        {
            d_compactMaxValue[loc_max] = t_max;
        }
    }
}

template <typename coord_t, typename real_t>
__global__ void setBoundaryMin(real_t* d_compactMinValue, coord_t* d_compactMinIndex, coord_t* d_MinScanResult, size_t SignalLength)
{
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.x;
    int pointsIdx = threadIdx.x;

    if ((pointsIdx == 0) && (d_MinScanResult[noiseIdx * SignalLength + SignalLength - 1] > 4))
    {
        real_t slope_min, t_min;

        coord_t storeLocation_min = d_MinScanResult[noiseIdx * SignalLength + pointsIdx];
        coord_t loc_min = noiseIdx * SignalLength + storeLocation_min;

        slope_min = (d_compactMinValue[loc_min + 2] - d_compactMinValue[loc_min + 1]) / (d_compactMinIndex[loc_min + 2] - d_compactMinIndex[loc_min + 1]);
        t_min = d_compactMinValue[loc_min + 1] - slope_min * (d_compactMinIndex[loc_min + 1] - d_compactMinIndex[loc_min]);

        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }

    if ((pointsIdx == 1) && (d_MinScanResult[noiseIdx * SignalLength + SignalLength - 1] > 4))
    {

        real_t slope_min, t_min;

        coord_t storeLocation_min = d_MinScanResult[noiseIdx * SignalLength + SignalLength - 1];
        coord_t loc_min = noiseIdx * SignalLength + storeLocation_min;

        slope_min = (d_compactMinValue[loc_min - 1] - d_compactMinValue[loc_min - 2]) / (d_compactMinIndex[loc_min - 1] - d_compactMinIndex[loc_min - 2]);
        t_min = d_compactMinValue[loc_min - 1] + slope_min * (d_compactMinIndex[loc_min] - d_compactMinIndex[loc_min - 1]);

        if (t_min < d_compactMinValue[loc_min])
        {
            d_compactMinValue[loc_min] = t_min;
        }
    }
}

template <typename real_t>
__global__ void preSetTridiagonalMatrix(real_t* d_upperDia, real_t* d_middleDia, real_t* d_lowerDia, real_t* d_right, size_t signalLnegth)
{
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = noiseIdx * signalLnegth + pointsIdx;

    if (pointsIdx < signalLnegth)
    {
        d_upperDia[idx] = 0;
        d_lowerDia[idx] = 0;
        d_right[idx] = 0;
        d_middleDia[idx] = 1;
    }

}

// for natural boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength) {
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = noiseIdx * SignalLength + pointsIdx;
    const int num_equation = d_num_extrema[noiseIdx];
    int idxForRight = noiseIdx * SignalLength + pointsIdx; // to compact the y value within one direction vector
    if (pointsIdx == 0)
    {
        d_middle_dia[idx] = 1;
        d_upper_dia[idx] = 0;
        d_lower_dia[idx] = 0;
        d_right_dia[idxForRight] = 0; // it used to be idx
    }
    if (pointsIdx == num_equation - 1)
    {
        d_middle_dia[idx] = 1;
        d_lower_dia[idx] = 0;
        d_upper_dia[idx] = 0;
        d_right_dia[idxForRight] = 0;
    }
    if (pointsIdx != 0 && pointsIdx < num_equation - 1)
    {
        d_middle_dia[idx] = 2 * (((d_extrema_x[idx] - d_extrema_x[idx - 1]) + (d_extrema_x[idx + 1] - d_extrema_x[idx])));
        d_upper_dia[idx] = d_extrema_x[idx + 1] - d_extrema_x[idx];
        d_lower_dia[idx] = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_right_dia[idxForRight] = 3 * ((d_extrema_y[idx + 1] - d_extrema_y[idx]) / (d_extrema_x[idx + 1] - d_extrema_x[idx]) -
            (d_extrema_y[idx] - d_extrema_y[idx - 1]) / (d_extrema_x[idx] - d_extrema_x[idx - 1]));
    }
}

// for not-a-knot boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup_nak(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength, size_t SignalDim, size_t NumDirVector) {
    int dirVecIdx = blockIdx.z;
    int signalDimIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = dirVecIdx * SignalDim * SignalLength + signalDimIdx * SignalLength + pointsIdx;
    const int num_equation = d_num_extrema[dirVecIdx] - 1;
    if (pointsIdx == 0)
    {
        real_t h0 = d_extrema_x[idx + 1] - d_extrema_x[idx];
        real_t h1 = d_extrema_x[idx + 2] - d_extrema_x[idx + 1];
        d_middle_dia[idx] = -h1 * h1 + h0 * h0;
        d_upper_dia[idx] = (h0 + h1) * h1 + 2 * h0 * (h0 + h1);
        d_lower_dia[idx] = 0; // fixed
        d_right_dia[idx] = 3 * (h0 / h1 * (d_extrema_y[idx + 2] - d_extrema_y[idx + 1]) - d_extrema_y[idx + 1] + d_extrema_y[idx]);
    }

    if (pointsIdx == num_equation)
    {
        real_t hn_2 = d_extrema_x[idx - 1] - d_extrema_x[idx - 2];
        real_t hn_1 = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_middle_dia[idx] = -hn_2 * hn_2 + hn_1 * hn_1;
        d_lower_dia[idx] = (hn_2 + hn_1) * hn_2 + hn_1 * 2 * (hn_2 + hn_1);
        d_upper_dia[idx] = 0; // fixed
        d_right_dia[idx] = 3 * ((d_extrema_y[idx] - d_extrema_y[idx - 1]) - hn_1 / hn_2 * (d_extrema_y[idx - 1] - d_extrema_y[idx - 2]));
    }
    if (pointsIdx != 0 && pointsIdx < num_equation)
    {
        d_middle_dia[idx] = 2 * (((d_extrema_x[idx] - d_extrema_x[idx - 1]) + (d_extrema_x[idx + 1] - d_extrema_x[idx])));
        d_upper_dia[idx] = d_extrema_x[idx + 1] - d_extrema_x[idx];
        d_lower_dia[idx] = d_extrema_x[idx] - d_extrema_x[idx - 1];
        d_right_dia[idx] = 3 * (d_extrema_y[idx + 1] - d_extrema_y[idx]) / (d_extrema_x[idx + 1] - d_extrema_x[idx]) - 3 * (d_extrema_y[idx] - d_extrema_y[idx - 1]) / (d_extrema_x[idx] - d_extrema_x[idx - 1]);
    }
}

template <typename coord_t, typename real_t>
__global__ void spline_coefficients(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* extrema_points_x, size_t SignalLength, coord_t* d_num_extrema, real_t* solution) {
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = noiseIdx * SignalLength + pointsIdx;

    const int num_equation = d_num_extrema[noiseIdx];
    int idxForSolution = noiseIdx * SignalLength + pointsIdx; // to index the compacted sulution matrix, it used to be idx

    if (pointsIdx < num_equation - 1)
    {
        real_t h = extrema_points_x[idx + 1] - extrema_points_x[idx];
        b[idx] = (a[idx + 1] - a[idx]) / h - h * (2 * solution[idxForSolution] + solution[idxForSolution + 1]) / 3;
        d[idx] = (solution[idxForSolution + 1] - solution[idxForSolution]) / (3 * h);
        c[idx] = solution[idxForSolution];
    }

}

template <typename coord_t, typename real_t>
__global__ void interpolate(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* d_envelopeIndex, real_t* d_envelopeValue, coord_t* d_extremaIndex, size_t SignalLength, coord_t* d_num_extrema) {
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = noiseIdx * SignalLength + pointsIdx;
    int idx2 = noiseIdx * SignalLength;

    int num_coefs = d_num_extrema[noiseIdx];
    size_t num_samples = SignalLength;
    if (pointsIdx < num_samples)// what the hell here? <= or <
    {
        //int i = 0;
        int coef_idx = 0;
        int low = 1;
        int high = num_coefs - 1;
        // binary search for coef index
        while (low <= high) {
            int mid = (low + high) / 2;
            if ((pointsIdx > d_extremaIndex[idx2 + mid - 1]) && (pointsIdx <= d_extremaIndex[idx2 + mid])) {
                coef_idx = mid - 1;
                break;
            }
            else if (pointsIdx < d_extremaIndex[idx2 + mid]) {
                high = mid - 1;
            }
            else {
                low = mid + 1;
            }
        }

        coord_t t = d_envelopeIndex[idx] - d_extremaIndex[idx2 + coef_idx];
        d_envelopeValue[idx] = a[idx2 + coef_idx] + (b[idx2 + coef_idx] + (c[idx2 + coef_idx] + d[idx2 + coef_idx] * t) * t) * t;
    }

}

template <typename coord_t, typename real_t>
__global__ void averageUppperLower(real_t* d_meanEnvelope, real_t* d_upperEnvelope, real_t* d_lowerEnvelope, size_t SignalLength, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min)
{
    //int dirVecIdx = blockIdx.z;
    //int signalDimIdx = blockIdx.y;
    int noiseIdx = blockIdx.y;
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = noiseIdx * SignalLength + pointsIdx;

    if ((d_num_extrema_max[noiseIdx] > 3) && (d_num_extrema_min[noiseIdx] > 3) && (pointsIdx < SignalLength))
    {
        d_meanEnvelope[idx] = (d_upperEnvelope[idx] + d_lowerEnvelope[idx]) / 2.0;
    }
}

template <typename real_t>
__global__ void produceSX(real_t* d_sxVector, real_t* d_upperEnvelope, real_t* d_lowerEnvelope, size_t SignalLength)
{
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;
    int idx = noiseIdx * SignalLength + pointsIdx;

    if (pointsIdx < SignalLength)
    {
        d_sxVector[idx] = abs((d_upperEnvelope[idx] + d_lowerEnvelope[idx]) / (d_upperEnvelope[idx] - d_lowerEnvelope[idx]));
    }
}

template <typename real_t>
__global__ void thresholdJudge(real_t* d_sxVector, real_t* d_channelMark, real_t threshold_1, real_t threshold_2, size_t signalLength)
{
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;

    if (pointsIdx < signalLength)
    {
        if (d_sxVector[noiseIdx * signalLength + pointsIdx] > threshold_2)
        {
            d_channelMark[noiseIdx] = 1;
        }

        if (d_sxVector[noiseIdx * signalLength + pointsIdx] > threshold_1)
        {
            d_sxVector[noiseIdx * signalLength + pointsIdx] = 1;
        }
        else
        {
            d_sxVector[noiseIdx * signalLength + pointsIdx] = 0;
        }
    }

}

template <typename coord_t, typename real_t>
__global__ void siftingCriterion(real_t* d_finishFlag, real_t* d_realizationMark, real_t* d_channelMeans, real_t* d_channelMark, real_t threshold_3, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min, size_t numNoise, size_t idxIter, size_t maxIter)
{
    int noiseIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (noiseIdx < numNoise)
    {
        int criterion1 = d_channelMeans[noiseIdx] > threshold_3 || d_channelMark[noiseIdx] && (d_num_extrema_max[noiseIdx] + d_num_extrema_min[noiseIdx]) > 6; // 2 + 4
        //int criterion2 = abs(d_num_zeroCrossPoints[noiseIdx] - d_num_extrema_max[noiseIdx] - d_num_extrema_min[noiseIdx] + 4) > 2; // 1 + 4
        //if (d1 || (d1 ^ d2))
        if (criterion1)
        {
            d_realizationMark[noiseIdx] = 1;
            if (d_finishFlag[noiseIdx] == 0 && idxIter == (maxIter - 1))
            {
                d_realizationMark[noiseIdx] = 0;
                d_finishFlag[noiseIdx] = 1;
            }
        }
        else
        {
            if (d_finishFlag[noiseIdx] == 0)
            {
                d_realizationMark[noiseIdx] = 0;
                d_finishFlag[noiseIdx] = 1;
            }
            else
            {
                d_realizationMark[noiseIdx] = 1;
            }
        }
    }
}

template <typename real_t>
__global__ void updateRealizations(real_t* d_realizationMark, real_t* currentWhiteNoiseModes, real_t* d_noisedSignal, real_t* d_meanEnvelope, size_t SignalLength, size_t numNoise, size_t j, size_t max_iter)
{
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;

    if (pointsIdx < SignalLength)
    {
        if ((d_realizationMark[noiseIdx] == 1))
        {
            d_noisedSignal[noiseIdx * SignalLength + pointsIdx] = d_noisedSignal[noiseIdx * SignalLength + pointsIdx] - d_meanEnvelope[noiseIdx * SignalLength + pointsIdx];
        }
        if ((d_realizationMark[noiseIdx] == 0))
        {
            currentWhiteNoiseModes[noiseIdx * SignalLength + pointsIdx] = d_noisedSignal[noiseIdx * SignalLength + pointsIdx];
        }
    }
}

template <typename real_t>
__global__ void checkBreak(real_t* d_finishFlag, int* whetherStopSifting, size_t numNoise)
{
    extern __shared__ int s_finishFlag[];
    int noiseIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = threadIdx.x;

    if (noiseIdx < blockDim.x)
    {
        s_finishFlag[index] = 1;
    }
    __syncthreads();

    if (noiseIdx < numNoise)
    {
        s_finishFlag[index] = d_finishFlag[noiseIdx] == 1;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (index < s)
            {
                s_finishFlag[index] &= s_finishFlag[index + s];
            }
            __syncthreads();
        }

        if (index == 0)
        {
            //whetherStopSifting[0] &= s_finishFlag[0];
            whetherStopSifting[0] = s_finishFlag[0];
        }
    }

}

template <typename real_t>
__global__ void produceResidue(real_t* d_noisedSignal, real_t* d_currentModes, real_t* d_residue, size_t SignalLength)
{
    int pointsIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int noiseIdx = blockIdx.y;
    if (pointsIdx < SignalLength)
    {
        d_residue[noiseIdx * SignalLength + pointsIdx] = d_noisedSignal[noiseIdx * SignalLength + pointsIdx] - d_currentModes[noiseIdx * SignalLength + pointsIdx];
    }
}

template <typename real_t>
__global__ void averageUpdateSignal(real_t* d_residue, real_t* d_forNext, real_t* d_IMFs, size_t numNoise, size_t SignalLength, size_t imfIdx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (SignalLength))
    {
        real_t tem = 0;
        for (int i = 0; i < numNoise; i++)
        {
            tem = tem + d_residue[i * SignalLength + idx];
        }
        d_IMFs[imfIdx * SignalLength + idx] = d_forNext[idx] - tem / real_t(numNoise);
        d_forNext[idx] = tem / real_t(numNoise);
    }
}

template <typename real_t>
__global__ void updateSignal(real_t* d_current, real_t* d_whiteNoise, size_t numNoise, size_t SignalLength)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (numNoise * SignalLength))
    {
        d_whiteNoise[idx] = d_whiteNoise[idx] - d_current[idx];
    }
}

double iceemdanS(size_t numNoise, size_t SignalLength, size_t num_IMFs, size_t max_iter, int* d_x, float* d_y, float* d_IMFs, float noiseStrength)
{
    //===============load data===============
    float* d_noisedSignal = NULL;
    cudaMalloc((void**)&d_noisedSignal, numNoise * SignalLength * sizeof(float));
    float* d_running = NULL;
    cudaMalloc((void**)&d_running, SignalLength * sizeof(float));

    //===============preparation for noise===============
    float* d_whiteNoise = NULL;
    cudaMalloc((void**)&d_whiteNoise, numNoise * SignalLength * sizeof(float));
    curandGenerator_t gen;
    float meanValue = 0.0;
    float stddev = 1.0;

    float* d_whiteNoiseModes = NULL;
    cudaMalloc((void**)&d_whiteNoiseModes, num_IMFs * numNoise * SignalLength * sizeof(float));
    float* d_current = NULL;
    cudaMalloc((void**)&d_current, numNoise * SignalLength * sizeof(float));

    float* d_currentModes = NULL;
    cudaMalloc((void**)&d_currentModes, numNoise * SignalLength * sizeof(float));

    float* d_channelMeans = NULL;
    cudaMalloc((void**)&d_channelMeans, numNoise * sizeof(float));
    float* d_channelVariance = NULL;
    cudaMalloc((void**)&d_channelVariance, numNoise * sizeof(float));

    float* d_singleChannelMean = NULL;
    cudaMalloc((void**)&d_singleChannelMean, sizeof(float));
    float* d_singleChannelVariance = NULL;
    cudaMalloc((void**)&d_singleChannelVariance, sizeof(float));

    //===============preparation for extreme points detection===============
    int* d_sparseFlag;
    cudaMalloc((void**)&d_sparseFlag, numNoise * SignalLength * sizeof(int));
    int* d_sparseZeroCrossFlag;
    cudaMalloc((void**)&d_sparseZeroCrossFlag, numNoise * SignalLength * sizeof(int));

    int* d_noisedSignalIndex = NULL;
    cudaMalloc((void**)&d_noisedSignalIndex, numNoise * SignalLength * sizeof(int));
    for (int i = 0; i < numNoise; i++)
    {
        int head = i * SignalLength;
        cudaMemcpy((d_noisedSignalIndex + head), d_x, SignalLength * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    //===============preparation for prefix sum===============
    int* d_ScanResult;
    cudaMalloc((void**)&d_ScanResult, numNoise * SignalLength * sizeof(int));

    void* d_temStorage = NULL;
    size_t tempStorageBytes = 128 * 1024 * 1024 * sizeof(float);
    cudaMalloc(&d_temStorage, tempStorageBytes);

    //===============preparation for extreme points select===============
    float* d_compactValue;
    int* d_compactIndex;
    int* d_num_extrema_max, * d_num_extrema_min; // *d_num_zeroCrossPoints;

    cudaMalloc((void**)&d_compactValue, numNoise * SignalLength * sizeof(float));
    cudaMalloc((void**)&d_compactIndex, numNoise * SignalLength * sizeof(int));
    cudaMalloc((void**)&d_num_extrema_max, numNoise * sizeof(int));
    cudaMalloc((void**)&d_num_extrema_min, numNoise * sizeof(int));

    //===============preparation for tridiagonal setting===============
    float* d_upperDia = NULL, * d_middleDia = NULL, * d_lowerDia = NULL, * d_right = NULL;

    cudaMalloc((void**)&d_upperDia, numNoise * SignalLength * sizeof(float));
    cudaMalloc((void**)&d_middleDia, numNoise * SignalLength * sizeof(float));
    cudaMalloc((void**)&d_lowerDia, numNoise * SignalLength * sizeof(float));
    cudaMalloc((void**)&d_right, numNoise * SignalLength * sizeof(float));

    //===============preparation for tridiagonal solving===============

    float* currentUpperDia = NULL;
    float* currentMiddleDia = NULL;
    float* currentLowerDia = NULL;
    //float* currentRightDia = NULL;
    float* currentSolution = NULL;
    int* h_num_extrema = (int*)malloc(numNoise * sizeof(int));

    cusparseHandle_t handle_sparse;
    cusparseCreate(&handle_sparse);
    size_t* buffer_size = (size_t*)malloc(sizeof(size_t));
    float* buffer = NULL;
    cudaMalloc(&buffer, 128 * 1024 * 1024 * sizeof(float));

    //===============preparation for interpolate values===============
    float* d_envelopeVauleMax = NULL, * d_envelopeVauleMin = NULL;
    cudaMalloc((void**)&d_envelopeVauleMax, numNoise * SignalLength * sizeof(float));
    cudaMalloc((void**)&d_envelopeVauleMin, numNoise * SignalLength * sizeof(float));

    //===============preparation for averaging upper and lower===============
    float* d_meanEnvelope = NULL;
    cudaMalloc((void**)&d_meanEnvelope, numNoise * SignalLength * sizeof(float));

    float* d_forNext = NULL;
    cudaMalloc((void**)&d_forNext, SignalLength * sizeof(float));

    float* d_residue = NULL;
    cudaMalloc((void**)&d_residue, numNoise * SignalLength * sizeof(float));

    //===============preparation for CUDA graphs===============
    cudaStream_t streamForGraphPreFix; // maybe we can capture on default stream?
    cudaStream_t streamForGraphTriSolve;

    cudaStreamCreate(&streamForGraphPreFix);
    cudaStreamCreate(&streamForGraphTriSolve);

    cudaGraph_t graphPreFix;
    cudaGraphExec_t graph_execPreFix;
    int isCapturedPrfFix = 0;

    cudaGraph_t graphTriSolve;
    cudaGraphExec_t graph_execTriSolve;
    cusparseSetStream(handle_sparse, streamForGraphTriSolve);
    int isCapturedTriSolve = 0;

    //===============generate white noise===============
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormal(gen, d_whiteNoise, numNoise * SignalLength, meanValue, stddev);

    //===============preparation for sifting stop===============
    float threshold_1 = 0.05; // sd
    float threshold_2 = 0.5; // sd2
    float threshold_3 = 0.05; // tol

    float* d_sxVector = NULL;
    cudaMalloc((void**)&d_sxVector, numNoise * SignalLength * sizeof(float));

    float* d_realizationMark = NULL;
    cudaMalloc((void**)&d_realizationMark, numNoise * sizeof(float));
    cudaMemset(d_realizationMark, 0, numNoise * sizeof(float));

    float* d_chanelMeansSiftingStop = NULL;
    cudaMalloc((void**)&d_chanelMeansSiftingStop, numNoise * sizeof(float));

    float* d_channelMark = NULL;
    cudaMalloc((void**)&d_channelMark, numNoise * sizeof(float));

    float* d_finishFlag = NULL;
    cudaMalloc((void**)&d_finishFlag, numNoise * sizeof(float));

    int* whetherStopSifting = NULL;
    cudaMallocHost((void**)&whetherStopSifting, sizeof(int), cudaHostAllocMapped);

    //===============preparation for grid and block dim===============
    dim3 blockDimShfl(256);
    size_t numThreads = (SignalLength / 30 + 1) * 32;
    dim3 gridDimShfl(numThreads / blockDimShfl.x + (numThreads % blockDimShfl.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimSelectExtrema(256);
    dim3 gridDimSelectExtrema(SignalLength / blockDimSelectExtrema.x + (SignalLength % blockDimSelectExtrema.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimSetBoundary(2);
    dim3 gridDimSetBoundary(numNoise);

    dim3 blockDimPreSetTri(256);
    dim3 gridDimPreSetTri(SignalLength / blockDimPreSetTri.x + (SignalLength % blockDimPreSetTri.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimTriSet(256);
    dim3 gridDimTriSet(SignalLength / blockDimTriSet.x + (SignalLength % blockDimTriSet.x == 0 ? 0 : 1), numNoise); // too much idle threads

    dim3 blockDimSplineCoe(256);
    dim3 gridDimSplineCoe(SignalLength / blockDimSplineCoe.x + (SignalLength % blockDimSplineCoe.x == 0 ? 0 : 1), numNoise); // too much idle threads

    dim3 blockDimInterpolate(256);
    dim3 gridDimInterpolate(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimMeanEnvelope(256);
    dim3 gridDimMeanEnvelope(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimProduceSX(256);
    dim3 gridDimProduceSX(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimThresholdJudge(256);
    dim3 gridDimThresholdJudge(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimMultiply(256);
    dim3 gridDimMultiply(numNoise / blockDimMultiply.x + (numNoise % blockDimMultiply.x == 0 ? 0 : 1));

    dim3 blockDimSiftingCriterion(256);
    dim3 gridDimSiftingCriterion(numNoise / blockDimSiftingCriterion.x + (numNoise % blockDimSiftingCriterion.x == 0 ? 0 : 1));

    dim3 blockDimUpdateRealizations(256);
    dim3 gridDimUpdateRealizations(SignalLength / blockDimUpdateRealizations.x + (SignalLength % blockDimUpdateRealizations.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimCheckBreak(256);
    dim3 gridDimCheckBreak(numNoise / blockDimCheckBreak.x + (numNoise % blockDimCheckBreak.x == 0 ? 0 : 1));

    dim3 blockDimAddNoiseSingleChannel(256);
    dim3 gridDimAddNoiseSingleChannel(SignalLength / blockDimAddNoiseSingleChannel.x + +(SignalLength % blockDimAddNoiseSingleChannel.x == 0 ? 0 : 1), 1);

    dim3 blockDimStandardize(256);
    dim3 gridDimStandardize(SignalLength / blockDimStandardize.x + +(SignalLength % blockDimStandardize.x == 0 ? 0 : 1), 1);

    dim3 blockDimAddNoise(256);
    dim3 gridDimAddNoise(SignalLength / blockDimAddNoise.x + +(SignalLength % blockDimAddNoise.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimProduceResidue(256);
    dim3 gridDimProduceResidue(SignalLength / blockDimProduceResidue.x + (SignalLength % blockDimProduceResidue.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimAverageUpdateSignal(256);
    dim3 gridDimAverageUpdateSignal(SignalLength / blockDimAverageUpdateSignal.x + (SignalLength % blockDimAverageUpdateSignal.x == 0 ? 0 : 1));

    dim3 blockDimProduceFirstIMF(256);
    dim3 gridDimProduceFirstIMF(SignalLength / blockDimProduceFirstIMF.x + (SignalLength % blockDimProduceFirstIMF.x == 0 ? 0 : 1));

    dim3 blockDimUpdateSignal(256);
    dim3 gridDimUpdateSignal(numNoise * SignalLength / blockDimUpdateSignal.x + (numNoise * SignalLength % blockDimUpdateSignal.x == 0 ? 0 : 1));

    //=============================================
    //         generate white noise modes
    //=============================================

    //==========replace the d_whiteNoise with a fixed one==========
    //char filePathInput[] = "E:\\Software\\Nvidia_Development\\Project\\CUDA_CEEMDAN\\CEEMDAN\\ceemd_paper_example\\white_noise.bin";
    //float* h_whiteNoise = (float*)malloc(numNoise * SignalLength * sizeof(float));
    //char* buf = (char*)malloc(numNoise * SignalLength * sizeof(float));
    //readBin(filePathInput, buf, numNoise* SignalLength * sizeof(float));
    //h_whiteNoise = (float*)buf;
    //cudaMemcpy(d_whiteNoise, h_whiteNoise, numNoise* SignalLength * sizeof(float), cudaMemcpyHostToDevice);
    //==========replace the d_whiteNoise with a fixed one==========

    for (size_t i = 0; i < num_IMFs - 1; ++i)
    {
        cudaMemcpy(d_current, d_whiteNoise, numNoise * SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(d_finishFlag, 0, numNoise * sizeof(float));
        *whetherStopSifting = 1;

        for (size_t j = 0; j < 2000; ++j) // default is 2000
        {
            //==================extreme points detection max============
            cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
            cudaMemset(d_channelMark, 0, numNoise * sizeof(float));
            find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

            //==================prefix scan max======================
            if (isCapturedPrfFix == 0)
            {
                cudaStreamBeginCapture(streamForGraphPreFix, cudaStreamCaptureModeGlobal);
                for (size_t k = 0; k < numNoise; k++)
                {
                    int offset = k * SignalLength;
                    cub::DeviceScan::ExclusiveSum(d_temStorage, tempStorageBytes, d_sparseFlag + offset, d_ScanResult + offset, SignalLength, streamForGraphPreFix);
                }
                cudaStreamEndCapture(streamForGraphPreFix, &graphPreFix);
                cudaDeviceSynchronize();
                cudaGraphInstantiate(&graph_execPreFix, graphPreFix, NULL, NULL, 0); // it was 0
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                isCapturedPrfFix = 1;
            }
            else
            {
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);
            }

            for (size_t k = 0; k < numNoise; k++)
            {
                int offset = k * SignalLength;
                cub::DeviceScan::ExclusiveSum(d_temStorage, tempStorageBytes, d_sparseFlag + offset, d_ScanResult + offset, SignalLength, streamForGraphPreFix);
            }

            //==================extreme points select max============
            select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                d_ScanResult, /*d_ScanResultZeroCross,*/ d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max /*d_num_zeroCrossPoints*/);

            setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength);

            //==================set up tridiagonal matrix max============
            preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            //==================solve tridiagonal matrix max============
            if (isCapturedTriSolve == 0)
            {
                cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                for (size_t k = 0; k < numNoise; k++)
                {
                    currentUpperDia = d_upperDia + k * SignalLength;
                    currentMiddleDia = d_middleDia + k * SignalLength;
                    currentLowerDia = d_lowerDia + k * SignalLength;
                    currentSolution = d_right + k * SignalLength;
					
                    cusparseSgtsv2_nopivot(handle_sparse, SignalLength * 0.5 + 2/*pow(0.9, i + 1)*/, 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * 0.5 + 2/*pow(0.9, i + 1)*/, buffer);
                }
                cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                cudaDeviceSynchronize();
                cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                isCapturedTriSolve = 1;
            }
            else
            {
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);
            }

            //==================compute spline coefficients max============
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

            //==================interpolate values max============
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

            //==================extreme points detection min============
            cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
            find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

            //==================prefix scan min============
            cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
            cudaStreamSynchronize(streamForGraphPreFix);

            //==================extreme points select min============
            select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

            setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength);

            //==================set up tridiagonal matrix min============
            preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            //==================solve tridiagonal matrix min============
            cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
            cudaStreamSynchronize(streamForGraphTriSolve);

            //==================compute spline coefficients min============
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

            //==================interpolate values min============
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

            //==================average upper and lower============
            averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

            //==================sifting stop critierion============
            // amp = abs(envmax-envmin)/2;
            // envmoy = (envmin+envmax)/2;
            // sx = abs(envmoy)./amp;
            // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
            produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

            //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
            //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
            thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

            //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
            meanS << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(float) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
            multiplyS << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

            // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
            siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, 2000); // deafault max iteration number is 2000

            //==================update each realization or export d_current as IMF============
            updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, (d_whiteNoiseModes + i * numNoise * SignalLength), d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

            //==================check break============
            checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
            cudaDeviceSynchronize();

            if (whetherStopSifting[0] == 1)
				break;
        }

        updateSignal << <gridDimUpdateSignal, blockDimUpdateSignal >> > ((d_whiteNoiseModes + i * numNoise * SignalLength), d_whiteNoise, numNoise, SignalLength);
    }
    cudaMemcpy(&d_whiteNoiseModes[(num_IMFs - 1) * numNoise * SignalLength], d_whiteNoise, numNoise * SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);

    //=============================================
    //                 CEEMDAN
    //=============================================

    //===============IMF loop===============
    cudaMemcpy(d_running, d_y, SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);
    double start_time = omp_get_wtime();
    for (size_t i = 0; i < num_IMFs - 1; ++i)
    {
        cudaMemset(d_finishFlag, 0, numNoise * sizeof(float));
        *whetherStopSifting = 1;

        if (i == 0)
        {
            //===============noise adding and signal standardization===============
            meanS <<<gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(float) >>> (d_running, SignalLength, 1, d_singleChannelMean);
            multiplyS <<<1, 1 >>> (d_singleChannelMean, d_singleChannelMean, (1.0 / SignalLength), 1);

            varianceS <<<gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(float) >>> (d_running, SignalLength, 1, d_singleChannelMean, d_singleChannelVariance);
            multiplyS <<<1, 1 >>> (d_singleChannelVariance, d_singleChannelVariance, (1.0 / (SignalLength - 1)), 1);

            meanS <<<gridDimAddNoise, blockDimAddNoise, numNoise * sizeof(float) >>> ((d_whiteNoiseModes + i * numNoise * SignalLength), SignalLength, numNoise, d_channelMeans);
            multiplyS <<<1, numNoise >>> (d_channelMeans, d_channelMeans, (1.0 / SignalLength), numNoise);

            varianceS <<<gridDimAddNoise, blockDimAddNoise, numNoise * sizeof(float) >>> ((d_whiteNoiseModes + i * numNoise * SignalLength), SignalLength, numNoise, d_channelMeans, d_channelVariance);
            multiplyS <<<1, numNoise >>> (d_channelVariance, d_channelVariance, (1.0 / (SignalLength - 1)), numNoise);

            standardizeRunning <<<gridDimStandardize, blockDimStandardize >> > (d_running, SignalLength, d_singleChannelVariance);
            addNoise <<<gridDimAddNoise, blockDimAddNoise >> > (d_noisedSignal, d_running, d_whiteNoiseModes + i * numNoise * SignalLength, SignalLength, noiseStrength, d_channelVariance);

            //===============sifting loop===============
            cudaMemcpy(d_current, d_noisedSignal, numNoise * SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);

            for (size_t j = 0; j < max_iter; ++j)
            {
                cudaMemset(d_channelMark, 0, numNoise * sizeof(float));

                //==================extreme points detection max============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan max======================
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select max============
                select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max);

                setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix max============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix max============
                if (isCapturedTriSolve == 0)
                {
                    cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                    for (size_t k = 0; k < numNoise; k++)
                    {
                        currentUpperDia = d_upperDia + k * SignalLength;
                        currentMiddleDia = d_middleDia + k * SignalLength;
                        currentLowerDia = d_lowerDia + k * SignalLength;
                        currentSolution = d_right + k * SignalLength;
                        cusparseSgtsv2_nopivot(handle_sparse, SignalLength * pow(0.5, i + 1), 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * pow(0.5, i + 1), buffer);
                    }
                    cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                    cudaDeviceSynchronize();
                    cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    isCapturedTriSolve = 1;
                }
                else
                {
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    cudaStreamSynchronize(streamForGraphTriSolve);
                }

                //==================compute spline coefficients max============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

                //==================interpolate values max============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

                //==================extreme points detection min============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan min============
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select min============
                select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

                setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix min============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix min============
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);

                //==================compute spline coefficients min============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

                //==================interpolate values min============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

                //==================average upper and lower============
                averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

                //==================sifting stop critierion============
                // amp = abs(envmax-envmin)/2;
                // envmoy = (envmin+envmax)/2;
                // sx = abs(envmoy)./amp;
                // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
                produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

                //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
                //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
                thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

                //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
                meanS << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(float) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
                multiplyS << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

                // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
                siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, max_iter);

                //==================update each realization or export d_current as IMF============
                updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, d_currentModes, d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

                //==================check break====================
                checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
                cudaDeviceSynchronize();

                if (whetherStopSifting[0] == 1)
                {
                    break;
                }
            }
            produceFirstIMF << <gridDimProduceFirstIMF, blockDimProduceFirstIMF >> > (d_IMFs, d_running, d_noisedSignal, d_currentModes, d_forNext, numNoise, SignalLength);
        }
        else
        {
            //===============noise adding===============
            meanS << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(float) >> > (d_forNext, SignalLength, 1, d_singleChannelMean);
            multiplyS << <1, 1 >> > (d_singleChannelMean, d_singleChannelMean, (1.0 / SignalLength), 1);

            varianceS << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(float) >> > (d_forNext, SignalLength, 1, d_singleChannelMean, d_singleChannelVariance);
            multiplyS << <1, 1 >> > (d_singleChannelVariance, d_singleChannelVariance, (1.0 / (SignalLength - 1)), 1);

            addNoise2 << <gridDimAddNoise, blockDimAddNoise >> > (d_noisedSignal, d_forNext, d_whiteNoiseModes + i * numNoise * SignalLength, SignalLength, noiseStrength, d_singleChannelVariance);

            //===============sifting loop===============
            cudaMemcpy(d_current, d_noisedSignal, numNoise * SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);
            for (size_t j = 0; j < max_iter; ++j)
            {
                cudaMemset(d_channelMark, 0, numNoise * sizeof(float));

                //==================extreme points detection max============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan max======================
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select max============
                select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max);

                setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix max============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix max============
                if (isCapturedTriSolve == 0)
                {
                    cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                    for (size_t k = 0; k < numNoise; k++)
                    {
                        currentUpperDia = d_upperDia + k * SignalLength;
                        currentMiddleDia = d_middleDia + k * SignalLength;
                        currentLowerDia = d_lowerDia + k * SignalLength;
                        currentSolution = d_right + k * SignalLength;
						
                        cusparseSgtsv2_nopivot(handle_sparse, SignalLength * pow(0.9, i + 1), 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * pow(0.9, i + 1), buffer);
                    }
                    cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                    cudaDeviceSynchronize();
                    cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    isCapturedTriSolve = 1;
                }
                else
                {
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    cudaStreamSynchronize(streamForGraphTriSolve);
                }

                //==================compute spline coefficients max============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

                //==================interpolate values max============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

                //==================extreme points detection min============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan min============
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select min============
                select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

                setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix min============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix min============     
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);

                //==================compute spline coefficients min============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

                //==================interpolate values min============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

                //==================average upper and lower============
                averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

                //==================sifting stop critierion============
                 // amp = abs(envmax-envmin)/2;
                 // envmoy = (envmin+envmax)/2;
                 // sx = abs(envmoy)./amp;
                 // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
                produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

                //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
                //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
                thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

                //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
                meanS << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(float) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
                multiplyS << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

                // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
                siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, max_iter);

                //==================update each realization or export d_current as IMF============
                updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, d_currentModes, d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

                //==================check break====================
                checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
                cudaDeviceSynchronize();

                if (whetherStopSifting[0] == 1)
                {
                    break;
                }
            }

            produceResidue << <gridDimProduceResidue, blockDimProduceResidue >> > (d_noisedSignal, d_currentModes, d_residue, SignalLength);
            averageUpdateSignal << <gridDimAverageUpdateSignal, blockDimAverageUpdateSignal >> > (d_residue, d_forNext, d_IMFs, numNoise, SignalLength, i);

            //=============================TEST============================
            //float* h_currentModes = (float*)malloc(SignalLength * sizeof(float));
            //cudaMemcpy(h_currentModes, d_currentModes, SignalLength * sizeof(float), cudaMemcpyDeviceToHost);
            //char test_file_h_currentModes[] = "./h_currentModes.bin";
            //writeBin(test_file_h_currentModes, (char*)h_currentModes, SignalLength * sizeof(float));
            //=============================TEST============================
        }
    }

    double end_time = omp_get_wtime();
    double diff = end_time - start_time;
    cudaMemcpy(&d_IMFs[(num_IMFs - 1) * SignalLength], d_forNext, SignalLength * sizeof(float), cudaMemcpyDeviceToDevice);

    //free all the CPU and GPU memory here
    cudaFree(d_noisedSignal);
    cudaFree(d_running);
    cudaFree(d_sparseFlag);
    cudaFree(d_ScanResult);
    cudaFree(d_compactValue);
    cudaFree(d_compactIndex);
    cudaFree(d_num_extrema_max);
    cudaFree(d_num_extrema_min);
    cudaFree(d_upperDia);
    cudaFree(d_middleDia);
    cudaFree(d_lowerDia);
    cudaFree(d_right);
    cudaFree(d_envelopeVauleMax);
    cudaFree(d_envelopeVauleMin);
    cudaFree(d_meanEnvelope);
    free(h_num_extrema);
    cudaFree(buffer);
    cudaFree(d_whiteNoise);
    cudaFree(d_whiteNoiseModes);
    cudaFree(d_current);
    cudaFree(d_currentModes);
    cudaFree(d_channelMeans);
    cudaFree(d_channelVariance);
    cudaFree(d_singleChannelMean);
    cudaFree(d_singleChannelVariance);
    cudaFree(d_sparseZeroCrossFlag);
    cudaFree(d_noisedSignalIndex);
    cudaFree(d_forNext);
    cudaFree(d_residue);
    cudaFree(d_sxVector);
    cudaFree(d_realizationMark);
    cudaFree(d_chanelMeansSiftingStop);
    cudaFree(d_channelMark);
    cudaFree(d_finishFlag);
    cudaFree(d_temStorage);

    return diff;
}

double iceemdanD(size_t numNoise, size_t SignalLength, size_t num_IMFs, size_t max_iter, int* d_x, double* d_y, double* d_IMFs, double noiseStrength)
{
    //===============load data===============
    double* d_noisedSignal = NULL;
    cudaMalloc((void**)&d_noisedSignal, numNoise * SignalLength * sizeof(double));
    double* d_running = NULL;
    cudaMalloc((void**)&d_running, SignalLength * sizeof(float));

    //===============preparation for noise===============
    double* d_whiteNoise = NULL;
    cudaMalloc((void**)&d_whiteNoise, numNoise * SignalLength * sizeof(double));
    curandGenerator_t gen;
    double meanValue = 0.0;
    double stddev = 1.0;

    double* d_whiteNoiseModes = NULL;
    cudaMalloc((void**)&d_whiteNoiseModes, num_IMFs * numNoise * SignalLength * sizeof(double));
    double* d_current = NULL;
    cudaMalloc((void**)&d_current, numNoise * SignalLength * sizeof(double));

    double* d_currentModes = NULL;
    cudaMalloc((void**)&d_currentModes, numNoise * SignalLength * sizeof(double));

    double* d_channelMeans = NULL;
    cudaMalloc((void**)&d_channelMeans, numNoise * sizeof(double));
    double* d_channelVariance = NULL;
    cudaMalloc((void**)&d_channelVariance, numNoise * sizeof(double));

    double* d_singleChannelMean = NULL;
    cudaMalloc((void**)&d_singleChannelMean, sizeof(double));
    double* d_singleChannelVariance = NULL;
    cudaMalloc((void**)&d_singleChannelVariance, sizeof(double));

    //===============preparation for extreme points detection===============
    int* d_sparseFlag;
    cudaMalloc((void**)&d_sparseFlag, numNoise * SignalLength * sizeof(int));
    int* d_sparseZeroCrossFlag;
    cudaMalloc((void**)&d_sparseZeroCrossFlag, numNoise * SignalLength * sizeof(int));

    int* d_noisedSignalIndex = NULL;
    cudaMalloc((void**)&d_noisedSignalIndex, numNoise * SignalLength * sizeof(int));
    for (int i = 0; i < numNoise; i++)
    {
        int head = i * SignalLength;
        cudaMemcpy((d_noisedSignalIndex + head), d_x, SignalLength * sizeof(int), cudaMemcpyDeviceToDevice);
    }

    //===============preparation for prefix sum===============
    int* d_ScanResult;
    cudaMalloc((void**)&d_ScanResult, numNoise * SignalLength * sizeof(int));

    void* d_temStorage = NULL;
    size_t tempStorageBytes = 128 * 1024 * 1024 * sizeof(double);
    cudaMalloc(&d_temStorage, tempStorageBytes);

    //===============preparation for extreme points select===============
    double* d_compactValue;
    int* d_compactIndex;
    int* d_num_extrema_max, * d_num_extrema_min; // *d_num_zeroCrossPoints;

    cudaMalloc((void**)&d_compactValue, numNoise * SignalLength * sizeof(double));
    cudaMalloc((void**)&d_compactIndex, numNoise * SignalLength * sizeof(int));
    cudaMalloc((void**)&d_num_extrema_max, numNoise * sizeof(int));
    cudaMalloc((void**)&d_num_extrema_min, numNoise * sizeof(int));

    //===============preparation for tridiagonal setting===============
    double* d_upperDia = NULL, * d_middleDia = NULL, * d_lowerDia = NULL, * d_right = NULL;

    cudaMalloc((void**)&d_upperDia, numNoise * SignalLength * sizeof(double));
    cudaMalloc((void**)&d_middleDia, numNoise * SignalLength * sizeof(double));
    cudaMalloc((void**)&d_lowerDia, numNoise * SignalLength * sizeof(double));
    cudaMalloc((void**)&d_right, numNoise * SignalLength * sizeof(double));

    //===============preparation for tridiagonal solving===============

    double* currentUpperDia = NULL;
    double* currentMiddleDia = NULL;
    double* currentLowerDia = NULL;
    //double* currentRightDia = NULL;
    double* currentSolution = NULL;
    int* h_num_extrema = (int*)malloc(numNoise * sizeof(int));

    cusparseHandle_t handle_sparse;
    cusparseCreate(&handle_sparse);
    size_t* buffer_size = (size_t*)malloc(sizeof(size_t));
    double* buffer = NULL;
    cudaMalloc(&buffer, 128 * 1024 * 1024 * sizeof(double));

    //===============preparation for interpolate values===============
    double* d_envelopeVauleMax = NULL, * d_envelopeVauleMin = NULL;
    cudaMalloc((void**)&d_envelopeVauleMax, numNoise * SignalLength * sizeof(double));
    cudaMalloc((void**)&d_envelopeVauleMin, numNoise * SignalLength * sizeof(double));

    //===============preparation for averaging upper and lower===============
    double* d_meanEnvelope = NULL;
    cudaMalloc((void**)&d_meanEnvelope, numNoise * SignalLength * sizeof(double));

    double* d_forNext = NULL;
    cudaMalloc((void**)&d_forNext, SignalLength * sizeof(double));

    double* d_residue = NULL;
    cudaMalloc((void**)&d_residue, numNoise * SignalLength * sizeof(double));

    //===============preparation for CUDA graphs===============
    cudaStream_t streamForGraphPreFix; // maybe we can capture on default stream?
    cudaStream_t streamForGraphTriSolve;

    cudaStreamCreate(&streamForGraphPreFix);
    cudaStreamCreate(&streamForGraphTriSolve);

    cudaGraph_t graphPreFix;
    cudaGraphExec_t graph_execPreFix;
    int isCapturedPrfFix = 0;

    cudaGraph_t graphTriSolve;
    cudaGraphExec_t graph_execTriSolve;
    cusparseSetStream(handle_sparse, streamForGraphTriSolve);
    int isCapturedTriSolve = 0;

    //===============generate white noise===============
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateNormalDouble(gen, d_whiteNoise, numNoise * SignalLength, meanValue, stddev);

    //===============preparation for sifting stop===============
    double threshold_1 = 0.05; // sd
    double threshold_2 = 0.5; // sd2
    double threshold_3 = 0.05; // tol

    double* d_sxVector = NULL;
    cudaMalloc((void**)&d_sxVector, numNoise * SignalLength * sizeof(double));

    double* d_realizationMark = NULL;
    cudaMalloc((void**)&d_realizationMark, numNoise * sizeof(double));
    cudaMemset(d_realizationMark, 0, numNoise * sizeof(double));

    double* d_chanelMeansSiftingStop = NULL;
    cudaMalloc((void**)&d_chanelMeansSiftingStop, numNoise * sizeof(double));

    double* d_channelMark = NULL;
    cudaMalloc((void**)&d_channelMark, numNoise * sizeof(double));

    double* d_finishFlag = NULL;
    cudaMalloc((void**)&d_finishFlag, numNoise * sizeof(double));

    int* whetherStopSifting = NULL;
    cudaMallocHost((void**)&whetherStopSifting, sizeof(int), cudaHostAllocMapped);

    //===============preparation for grid and block dim===============
    dim3 blockDimShfl(256);
    size_t numThreads = (SignalLength / 30 + 1) * 32;
    dim3 gridDimShfl(numThreads / blockDimShfl.x + (numThreads % blockDimShfl.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimSelectExtrema(256);
    dim3 gridDimSelectExtrema(SignalLength / blockDimSelectExtrema.x + (SignalLength % blockDimSelectExtrema.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimSetBoundary(2);
    dim3 gridDimSetBoundary(numNoise);

    dim3 blockDimPreSetTri(256);
    dim3 gridDimPreSetTri(SignalLength / blockDimPreSetTri.x + (SignalLength % blockDimPreSetTri.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimTriSet(256);
    dim3 gridDimTriSet(SignalLength / blockDimTriSet.x + (SignalLength % blockDimTriSet.x == 0 ? 0 : 1), numNoise); // too much idle threads

    dim3 blockDimSplineCoe(256);
    dim3 gridDimSplineCoe(SignalLength / blockDimSplineCoe.x + (SignalLength % blockDimSplineCoe.x == 0 ? 0 : 1), numNoise); // too much idle threads

    dim3 blockDimInterpolate(256);
    dim3 gridDimInterpolate(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimMeanEnvelope(256);
    dim3 gridDimMeanEnvelope(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimProduceSX(256);
    dim3 gridDimProduceSX(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimThresholdJudge(256);
    dim3 gridDimThresholdJudge(SignalLength / blockDimInterpolate.x + (SignalLength % blockDimInterpolate.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimMultiply(256);
    dim3 gridDimMultiply(numNoise / blockDimMultiply.x + (numNoise % blockDimMultiply.x == 0 ? 0 : 1));

    dim3 blockDimSiftingCriterion(256);
    dim3 gridDimSiftingCriterion(numNoise / blockDimSiftingCriterion.x + (numNoise % blockDimSiftingCriterion.x == 0 ? 0 : 1));

    dim3 blockDimUpdateRealizations(256);
    dim3 gridDimUpdateRealizations(SignalLength / blockDimUpdateRealizations.x + (SignalLength % blockDimUpdateRealizations.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimCheckBreak(256);
    dim3 gridDimCheckBreak(numNoise / blockDimCheckBreak.x + (numNoise % blockDimCheckBreak.x == 0 ? 0 : 1));

    dim3 blockDimAddNoiseSingleChannel(256);
    dim3 gridDimAddNoiseSingleChannel(SignalLength / blockDimAddNoiseSingleChannel.x + +(SignalLength % blockDimAddNoiseSingleChannel.x == 0 ? 0 : 1), 1);

    dim3 blockDimStandardize(256);
    dim3 gridDimStandardize(SignalLength / blockDimStandardize.x + +(SignalLength % blockDimStandardize.x == 0 ? 0 : 1), 1);

    dim3 blockDimAddNoise(256);
    dim3 gridDimAddNoise(SignalLength / blockDimAddNoise.x + +(SignalLength % blockDimAddNoise.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimProduceResidue(256);
    dim3 gridDimProduceResidue(SignalLength / blockDimProduceResidue.x + (SignalLength % blockDimProduceResidue.x == 0 ? 0 : 1), numNoise);

    dim3 blockDimAverageUpdateSignal(256);
    dim3 gridDimAverageUpdateSignal(SignalLength / blockDimAverageUpdateSignal.x + (SignalLength % blockDimAverageUpdateSignal.x == 0 ? 0 : 1));

    dim3 blockDimProduceFirstIMF(256);
    dim3 gridDimProduceFirstIMF(SignalLength / blockDimProduceFirstIMF.x + (SignalLength % blockDimProduceFirstIMF.x == 0 ? 0 : 1));

    dim3 blockDimUpdateSignal(256);
    dim3 gridDimUpdateSignal(numNoise * SignalLength / blockDimUpdateSignal.x + (numNoise * SignalLength % blockDimUpdateSignal.x == 0 ? 0 : 1));

    //=============================================
    //         generate white noise modes
    //=============================================

    //==========replace the d_whiteNoise with a fixed one==========
    //char filePathInput[] = "E:\\Software\\Nvidia_Development\\Project\\CUDA_CEEMDAN\\CEEMDAN\\ceemd_paper_example\\white_noise.bin";
    //double* h_whiteNoise = (double*)malloc(numNoise * SignalLength * sizeof(double));
    //char* buf = (char*)malloc(numNoise * SignalLength * sizeof(double));
    //readBin(filePathInput, buf, numNoise* SignalLength * sizeof(double));
    //h_whiteNoise = (double*)buf;
    //cudaMemcpy(d_whiteNoise, h_whiteNoise, numNoise* SignalLength * sizeof(double), cudaMemcpyHostToDevice);
    //==========replace the d_whiteNoise with a fixed one==========

    for (size_t i = 0; i < num_IMFs - 1; ++i)
    {
        cudaMemcpy(d_current, d_whiteNoise, numNoise * SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemset(d_finishFlag, 0, numNoise * sizeof(double));
        *whetherStopSifting = 1;

        for (size_t j = 0; j < 2000; ++j) // default is 2000
        {
            //==================extreme points detection max============
            cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
            cudaMemset(d_channelMark, 0, numNoise * sizeof(double));
            find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

            //==================prefix scan max======================
            if (isCapturedPrfFix == 0)
            {
                cudaStreamBeginCapture(streamForGraphPreFix, cudaStreamCaptureModeGlobal);
                for (size_t k = 0; k < numNoise; k++)
                {
                    int offset = k * SignalLength;
                    cub::DeviceScan::ExclusiveSum(d_temStorage, tempStorageBytes, d_sparseFlag + offset, d_ScanResult + offset, SignalLength, streamForGraphPreFix);
                }
                cudaStreamEndCapture(streamForGraphPreFix, &graphPreFix);
                cudaDeviceSynchronize();
                cudaGraphInstantiate(&graph_execPreFix, graphPreFix, NULL, NULL, 0); // it was 0
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                isCapturedPrfFix = 1;
            }
            else
            {
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);
            }

            for (size_t k = 0; k < numNoise; k++)
            {
                int offset = k * SignalLength;
                cub::DeviceScan::ExclusiveSum(d_temStorage, tempStorageBytes, d_sparseFlag + offset, d_ScanResult + offset, SignalLength, streamForGraphPreFix);
            }

            //==================extreme points select max============
            select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                d_ScanResult, /*d_ScanResultZeroCross,*/ d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max /*d_num_zeroCrossPoints*/);

            setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength);

            //==================set up tridiagonal matrix max============
            preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            //==================solve tridiagonal matrix max============
            if (isCapturedTriSolve == 0)
            {
                cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                for (size_t k = 0; k < numNoise; k++)
                {
                    currentUpperDia = d_upperDia + k * SignalLength;
                    currentMiddleDia = d_middleDia + k * SignalLength;
                    currentLowerDia = d_lowerDia + k * SignalLength;
                    currentSolution = d_right + k * SignalLength;

                    cusparseDgtsv2_nopivot(handle_sparse, SignalLength * 0.5 + 2/*pow(0.9, i + 1)*/, 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * 0.5 + 2/*pow(0.9, i + 1)*/, buffer);
                }
                cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                cudaDeviceSynchronize();
                cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                isCapturedTriSolve = 1;
            }
            else
            {
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);
            }

            //==================compute spline coefficients max============
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

            //==================interpolate values max============
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

            //==================extreme points detection min============
            cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
            find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

            //==================prefix scan min============
            cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
            cudaStreamSynchronize(streamForGraphPreFix);

            //==================extreme points select min============
            select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

            setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                d_ScanResult, SignalLength);

            //==================set up tridiagonal matrix min============
            preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

            //==================solve tridiagonal matrix min============
            cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
            cudaStreamSynchronize(streamForGraphTriSolve);

            //==================compute spline coefficients min============
            spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

            //==================interpolate values min============
            interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

            //==================average upper and lower============
            averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

            //==================sifting stop critierion============
            // amp = abs(envmax-envmin)/2;
            // envmoy = (envmin+envmax)/2;
            // sx = abs(envmoy)./amp;
            // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
            produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

            //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
            //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
            thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

            //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
            meanD << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(double) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
            multiplyD << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

            // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
            siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, 2000); // deafault max iteration number is 2000

            //==================update each realization or export d_current as IMF============
            updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, (d_whiteNoiseModes + i * numNoise * SignalLength), d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

            //==================check break============
            checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
            cudaDeviceSynchronize();

            if (whetherStopSifting[0] == 1)
                break;
        }

        updateSignal << <gridDimUpdateSignal, blockDimUpdateSignal >> > ((d_whiteNoiseModes + i * numNoise * SignalLength), d_whiteNoise, numNoise, SignalLength);
    }
    cudaMemcpy(&d_whiteNoiseModes[(num_IMFs - 1) * numNoise * SignalLength], d_whiteNoise, numNoise * SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);

    //=============================================
    //                 CEEMDAN
    //=============================================

    //===============IMF loop===============
    cudaMemcpy(d_running, d_y, SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);
    double start_time = omp_get_wtime();
    for (size_t i = 0; i < num_IMFs - 1; ++i)
    {
        cudaMemset(d_finishFlag, 0, numNoise * sizeof(double));
        *whetherStopSifting = 1;

        if (i == 0)
        {
            //===============noise adding and signal standardization===============
            meanD << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(double) >> > (d_running, SignalLength, 1, d_singleChannelMean);
            multiplyD << <1, 1 >> > (d_singleChannelMean, d_singleChannelMean, (1.0 / SignalLength), 1);

            varianceD << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(double) >> > (d_running, SignalLength, 1, d_singleChannelMean, d_singleChannelVariance);
            multiplyD << <1, 1 >> > (d_singleChannelVariance, d_singleChannelVariance, (1.0 / (SignalLength - 1)), 1);

            meanD << <gridDimAddNoise, blockDimAddNoise, numNoise * sizeof(double) >> > ((d_whiteNoiseModes + i * numNoise * SignalLength), SignalLength, numNoise, d_channelMeans);
            multiplyD << <1, numNoise >> > (d_channelMeans, d_channelMeans, (1.0 / SignalLength), numNoise);

            varianceD << <gridDimAddNoise, blockDimAddNoise, numNoise * sizeof(double) >> > ((d_whiteNoiseModes + i * numNoise * SignalLength), SignalLength, numNoise, d_channelMeans, d_channelVariance);
            multiplyD << <1, numNoise >> > (d_channelVariance, d_channelVariance, (1.0 / (SignalLength - 1)), numNoise);

            standardizeRunning << <gridDimStandardize, blockDimStandardize >> > (d_running, SignalLength, d_singleChannelVariance);
            addNoise << <gridDimAddNoise, blockDimAddNoise >> > (d_noisedSignal, d_running, d_whiteNoiseModes + i * numNoise * SignalLength, SignalLength, noiseStrength, d_channelVariance);

            //===============sifting loop===============
            cudaMemcpy(d_current, d_noisedSignal, numNoise * SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);

            for (size_t j = 0; j < max_iter; ++j)
            {
                cudaMemset(d_channelMark, 0, numNoise * sizeof(double));

                //==================extreme points detection max============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan max======================
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select max============
                select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max);

                setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix max============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix max============
                if (isCapturedTriSolve == 0)
                {
                    cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                    for (size_t k = 0; k < numNoise; k++)
                    {
                        currentUpperDia = d_upperDia + k * SignalLength;
                        currentMiddleDia = d_middleDia + k * SignalLength;
                        currentLowerDia = d_lowerDia + k * SignalLength;
                        currentSolution = d_right + k * SignalLength;
                        cusparseDgtsv2_nopivot(handle_sparse, SignalLength * pow(0.5, i + 1), 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * pow(0.5, i + 1), buffer);
                    }
                    cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                    cudaDeviceSynchronize();
                    cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    isCapturedTriSolve = 1;
                }
                else
                {
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    cudaStreamSynchronize(streamForGraphTriSolve);
                }

                //==================compute spline coefficients max============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

                //==================interpolate values max============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

                //==================extreme points detection min============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan min============
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select min============
                select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

                setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix min============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix min============
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);

                //==================compute spline coefficients min============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

                //==================interpolate values min============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

                //==================average upper and lower============
                averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

                //==================sifting stop critierion============
                // amp = abs(envmax-envmin)/2;
                // envmoy = (envmin+envmax)/2;
                // sx = abs(envmoy)./amp;
                // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
                produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

                //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
                //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
                thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

                //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
                meanD << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(double) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
                multiplyD << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

                // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
                siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, max_iter);

                //==================update each realization or export d_current as IMF============
                updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, d_currentModes, d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

                //==================check break====================
                checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
                cudaDeviceSynchronize();

                if (whetherStopSifting[0] == 1)
                {
                    break;
                }
            }
            produceFirstIMF << <gridDimProduceFirstIMF, blockDimProduceFirstIMF >> > (d_IMFs, d_running, d_noisedSignal, d_currentModes, d_forNext, numNoise, SignalLength);
        }
        else
        {
            //===============noise adding===============
            meanD << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(double) >> > (d_forNext, SignalLength, 1, d_singleChannelMean);
            multiplyD << <1, 1 >> > (d_singleChannelMean, d_singleChannelMean, (1.0 / SignalLength), 1);

            varianceD << <gridDimAddNoiseSingleChannel, blockDimAddNoiseSingleChannel, 1 * sizeof(double) >> > (d_forNext, SignalLength, 1, d_singleChannelMean, d_singleChannelVariance);
            multiplyD << <1, 1 >> > (d_singleChannelVariance, d_singleChannelVariance, (1.0 / (SignalLength - 1)), 1);

            addNoise2 << <gridDimAddNoise, blockDimAddNoise >> > (d_noisedSignal, d_forNext, d_whiteNoiseModes + i * numNoise * SignalLength, SignalLength, noiseStrength, d_singleChannelVariance);

            //===============sifting loop===============
            cudaMemcpy(d_current, d_noisedSignal, numNoise * SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);
            for (size_t j = 0; j < max_iter; ++j)
            {
                cudaMemset(d_channelMark, 0, numNoise * sizeof(double));

                //==================extreme points detection max============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_max << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan max======================
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select max============
                select_extrema_max << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_max);

                setBoundaryMax << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix max============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_max, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix max============
                if (isCapturedTriSolve == 0)
                {
                    cudaStreamBeginCapture(streamForGraphTriSolve, cudaStreamCaptureModeGlobal);
                    for (size_t k = 0; k < numNoise; k++)
                    {
                        currentUpperDia = d_upperDia + k * SignalLength;
                        currentMiddleDia = d_middleDia + k * SignalLength;
                        currentLowerDia = d_lowerDia + k * SignalLength;
                        currentSolution = d_right + k * SignalLength;

                        cusparseDgtsv2_nopivot(handle_sparse, SignalLength * pow(0.9, i + 1), 1, currentLowerDia, currentMiddleDia, currentUpperDia, currentSolution, SignalLength * pow(0.9, i + 1), buffer);
                    }
                    cudaStreamEndCapture(streamForGraphTriSolve, &graphTriSolve);
                    cudaDeviceSynchronize();
                    cudaGraphInstantiate(&graph_execTriSolve, graphTriSolve, NULL, NULL, 0); // it was 0
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    isCapturedTriSolve = 1;
                }
                else
                {
                    cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                    cudaStreamSynchronize(streamForGraphTriSolve);
                }

                //==================compute spline coefficients max============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_max, d_right);

                //==================interpolate values max============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMax, d_compactIndex, SignalLength, d_num_extrema_max);

                //==================extreme points detection min============
                cudaMemset(d_sparseFlag, 0, numNoise * SignalLength * sizeof(int));
                find_extrema_shfl_min << <gridDimShfl, blockDimShfl >> > (d_noisedSignalIndex, d_current, d_sparseFlag, SignalLength);

                //==================prefix scan min============
                cudaGraphLaunch(graph_execPreFix, streamForGraphPreFix);
                cudaStreamSynchronize(streamForGraphPreFix);

                //==================extreme points select min============
                select_extrema_min << <gridDimSelectExtrema, blockDimSelectExtrema >> > (d_sparseFlag, d_current, d_noisedSignalIndex,
                    d_ScanResult, d_compactValue, d_compactIndex, SignalLength, d_num_extrema_min);

                setBoundaryMin << <gridDimSetBoundary, blockDimSetBoundary >> > (d_compactValue, d_compactIndex,
                    d_ScanResult, SignalLength);

                //==================set up tridiagonal matrix min============
                preSetTridiagonalMatrix << <gridDimPreSetTri, blockDimPreSetTri >> > (d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                tridiagonal_setup << <gridDimTriSet, blockDimTriSet >> > (d_num_extrema_min, d_compactIndex, d_compactValue,
                    d_upperDia, d_middleDia, d_lowerDia, d_right, SignalLength);

                //==================solve tridiagonal matrix min============     
                cudaGraphLaunch(graph_execTriSolve, streamForGraphTriSolve);
                cudaStreamSynchronize(streamForGraphTriSolve);

                //==================compute spline coefficients min============
                spline_coefficients << <gridDimSplineCoe, blockDimSplineCoe >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_compactIndex, SignalLength, d_num_extrema_min, d_right);

                //==================interpolate values min============
                interpolate << <gridDimInterpolate, blockDimInterpolate >> > (d_compactValue, d_upperDia, d_middleDia, d_lowerDia, d_noisedSignalIndex, d_envelopeVauleMin, d_compactIndex, SignalLength, d_num_extrema_min);

                //==================average upper and lower============
                averageUppperLower << <gridDimMeanEnvelope, blockDimMeanEnvelope >> > (d_meanEnvelope, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength, d_num_extrema_max, d_num_extrema_min);

                //==================sifting stop critierion============
                 // amp = abs(envmax-envmin)/2;
                 // envmoy = (envmin+envmax)/2;
                 // sx = abs(envmoy)./amp;
                 // so, sx = abs(envmin+envmax) / abs(envmin-envmax)
                produceSX << <gridDimProduceSX, blockDimProduceSX >> > (d_sxVector, d_envelopeVauleMax, d_envelopeVauleMin, SignalLength);

                //sx > sd; produce a boolean matrix (samples * channels) and store back in d_sxVector 
                //any(sx > sd2); produce a boolean vector (1 * channels) and store in d_channelMark
                thresholdJudge << <gridDimThresholdJudge, blockDimThresholdJudge >> > (d_sxVector, d_channelMark, threshold_1, threshold_2, SignalLength);

                //mean(sx > sd); calculate the mean of the bollean matrix for each channel and store in d_chanelMeansSiftingStop
                meanD << <gridDimMeanEnvelope, blockDimMeanEnvelope, numNoise * sizeof(double) >> > (d_sxVector, SignalLength, numNoise, d_chanelMeansSiftingStop);
                multiplyD << <gridDimMultiply, blockDimMultiply >> > (d_chanelMeansSiftingStop, d_chanelMeansSiftingStop, (1.0 / SignalLength), numNoise);

                // (mean(sx > sd) > tol | any(sx > sd2)) & (all(nem > 2)); store the result in d_realizationMark
                siftingCriterion << <gridDimSiftingCriterion, blockDimSiftingCriterion >> > (d_finishFlag, d_realizationMark, d_chanelMeansSiftingStop, d_channelMark, threshold_3, d_num_extrema_max, d_num_extrema_min, numNoise, j, max_iter);

                //==================update each realization or export d_current as IMF============
                updateRealizations << <gridDimUpdateRealizations, blockDimUpdateRealizations >> > (d_realizationMark, d_currentModes, d_current, d_meanEnvelope, SignalLength, numNoise, j, max_iter);

                //==================check break====================
                checkBreak << <gridDimCheckBreak, blockDimCheckBreak, 256 * sizeof(int) >> > (d_finishFlag, whetherStopSifting, numNoise);
                cudaDeviceSynchronize();

                if (whetherStopSifting[0] == 1)
                {
                    break;
                }
            }

            produceResidue << <gridDimProduceResidue, blockDimProduceResidue >> > (d_noisedSignal, d_currentModes, d_residue, SignalLength);
            averageUpdateSignal << <gridDimAverageUpdateSignal, blockDimAverageUpdateSignal >> > (d_residue, d_forNext, d_IMFs, numNoise, SignalLength, i);

            //=============================TEST============================
            //double* h_currentModes = (double*)malloc(SignalLength * sizeof(double));
            //cudaMemcpy(h_currentModes, d_currentModes, SignalLength * sizeof(double), cudaMemcpyDeviceToHost);
            //char test_file_h_currentModes[] = "./h_currentModes.bin";
            //writeBin(test_file_h_currentModes, (char*)h_currentModes, SignalLength * sizeof(double));
            //=============================TEST============================
        }
    }

    double end_time = omp_get_wtime();
    double diff = end_time - start_time;
    cudaMemcpy(&d_IMFs[(num_IMFs - 1) * SignalLength], d_forNext, SignalLength * sizeof(double), cudaMemcpyDeviceToDevice);

    //free all the CPU and GPU memory here
    cudaFree(d_noisedSignal);
    cudaFree(d_running);
    cudaFree(d_sparseFlag);
    cudaFree(d_ScanResult);
    cudaFree(d_compactValue);
    cudaFree(d_compactIndex);
    cudaFree(d_num_extrema_max);
    cudaFree(d_num_extrema_min);
    cudaFree(d_upperDia);
    cudaFree(d_middleDia);
    cudaFree(d_lowerDia);
    cudaFree(d_right);
    cudaFree(d_envelopeVauleMax);
    cudaFree(d_envelopeVauleMin);
    cudaFree(d_meanEnvelope);
    free(h_num_extrema);
    cudaFree(buffer);
    cudaFree(d_whiteNoise);
    cudaFree(d_whiteNoiseModes);
    cudaFree(d_current);
    cudaFree(d_currentModes);
    cudaFree(d_channelMeans);
    cudaFree(d_channelVariance);
    cudaFree(d_singleChannelMean);
    cudaFree(d_singleChannelVariance);
    cudaFree(d_sparseZeroCrossFlag);
    cudaFree(d_noisedSignalIndex);
    cudaFree(d_forNext);
    cudaFree(d_residue);
    cudaFree(d_sxVector);
    cudaFree(d_realizationMark);
    cudaFree(d_chanelMeansSiftingStop);
    cudaFree(d_channelMark);
    cudaFree(d_finishFlag);
    cudaFree(d_temStorage);

    return diff;
}
