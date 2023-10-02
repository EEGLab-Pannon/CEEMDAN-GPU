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

/* Compile the code on linux as follows after changing the -arch parameter to mathc your target device */
//nvcc -arch=sm_70 -Xcompiler -fopenmp -lcublas -lcusparse -lcurand ./demo.cu -o CUDA_CEEMDAN

template <typename real_t>
__global__ void produceFirstIMF(real_t* d_IMFs, real_t* d_running, real_t* d_noisedSignal, real_t* d_currentModes, real_t* d_forNext, size_t numNoise, size_t signalLength);

template <typename real_t>
__global__ void standardizeRunning(real_t* d_running, size_t SignalLength, real_t* d_singleChannelVariance);

template <typename real_t>
__global__ void addNoise(real_t* d_noisedSignal, real_t* d_running, real_t* d_whiteNoiseModes, size_t SignalLength, real_t noiseStrength, real_t* d_channelVariance);

template <typename real_t>
__global__ void addNoise2(real_t* d_noisedSignal, real_t* d_running, real_t* d_whiteNoiseModes, size_t SignalLength, real_t noiseStrength, real_t* d_singleChannelVariance);

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_max(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMaxFlag, size_t SignalLength);

template <typename coord_t, typename real_t>
__global__ void find_extrema_shfl_min(const coord_t* d_multiChannelIndex, const real_t* d_ProjectSignals, coord_t* d_sparseMinFlag, size_t SignalLength);

template <typename coord_t, typename real_t>
__global__ void select_extrema_max(coord_t* d_sparseMaxFlag, real_t* d_noisedSignal, coord_t* d_noisedSignalIndex, coord_t* d_MaxScanResult, real_t* d_compactMaxValue,
    coord_t* d_compactMaxIndex, size_t SignalLength, coord_t* d_num_extrema_max);

template <typename coord_t, typename real_t>
__global__ void select_extrema_min(coord_t* d_sparseMinFlag, real_t* d_noisedSignal, coord_t* d_noisedSignalIndex, coord_t* d_MinScanResult, real_t* d_compactMinValue,
    coord_t* d_compactMinIndex, size_t SignalLength, coord_t* d_num_extrema_min);

template <typename coord_t, typename real_t>
__global__ void setBoundaryMax(real_t* d_compactMaxValue, coord_t* d_compactMaxIndex, coord_t* d_MaxScanResult, size_t SignalLength);

template <typename coord_t, typename real_t>
__global__ void setBoundaryMin(real_t* d_compactMinValue, coord_t* d_compactMinIndex, coord_t* d_MinScanResult, size_t SignalLength);

template <typename real_t>
__global__ void preSetTridiagonalMatrix(real_t* d_upperDia, real_t* d_middleDia, real_t* d_lowerDia, real_t* d_right, size_t signalLnegth);

// for natural boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength);

// for not-a-knot boundary conditions
template <typename coord_t, typename real_t>
__global__ void tridiagonal_setup_nak(coord_t* d_num_extrema, coord_t* d_extrema_x, real_t* d_extrema_y, real_t* d_upper_dia, real_t* d_middle_dia, real_t* d_lower_dia, real_t* d_right_dia, size_t SignalLength, size_t SignalDim, size_t NumDirVector);

template <typename coord_t, typename real_t>
__global__ void spline_coefficients(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* extrema_points_x, size_t SignalLength, coord_t* d_num_extrema, real_t* solution);

template <typename coord_t, typename real_t>
__global__ void interpolate(const real_t* a, real_t* b, real_t* c, real_t* d, coord_t* d_envelopeIndex, real_t* d_envelopeValue, coord_t* d_extremaIndex, size_t SignalLength, coord_t* d_num_extrema);

template <typename coord_t, typename real_t>
__global__ void averageUppperLower(real_t* d_meanEnvelope, real_t* d_upperEnvelope, real_t* d_lowerEnvelope, size_t SignalLength, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min);

template <typename real_t>
__global__ void produceSX(real_t* d_sxVector, real_t* d_upperEnvelope, real_t* d_lowerEnvelope, size_t SignalLength);

template <typename real_t>
__global__ void thresholdJudge(real_t* d_sxVector, real_t* d_channelMark, real_t threshold_1, real_t threshold_2, size_t signalLength);

template <typename coord_t, typename real_t>
__global__ void siftingCriterion(real_t* d_finishFlag, real_t* d_realizationMark, real_t* d_channelMeans, real_t* d_channelMark, real_t threshold_3, coord_t* d_num_extrema_max, coord_t* d_num_extrema_min, size_t numNoise, size_t idxIter, size_t maxIter);

template <typename real_t>
__global__ void updateRealizations(real_t* d_realizationMark, real_t* currentWhiteNoiseModes, real_t* d_noisedSignal, real_t* d_meanEnvelope, size_t SignalLength, size_t numNoise, size_t j, size_t max_iter);

template <typename real_t>
__global__ void checkBreak(real_t* d_finishFlag, int* whetherStopSifting, size_t numNoise);

template <typename real_t>
__global__ void produceResidue(real_t* d_noisedSignal, real_t* d_currentModes, real_t* d_residue, size_t SignalLength);

template <typename real_t>
__global__ void averageUpdateSignal(real_t* d_residue, real_t* d_forNext, real_t* d_IMFs, size_t numNoise, size_t SignalLength, size_t imfIdx);

template <typename real_t>
__global__ void updateSignal(real_t* d_current, real_t* d_whiteNoise, size_t numNoise, size_t SignalLength);

template <typename coord_t, typename real_t>
double ceemdan(size_t numNoise, size_t SignalLength, size_t num_IMFs, size_t max_iter, coord_t* d_x, real_t* d_y, real_t* d_IMFs, real_t noiseStrength);
