/*
MIT License

Copyright (c) 2023 Electrical Brain Imaging Lab, University of Pannonia, Veszprem, Hungary

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

__device__ float warpReduceSumS(float val);

__device__ float blockReduceSumS(float val);

__global__ void multiplyS(float* input, float* output, float alpha, int N);

__global__ void meanS(float* in, const int N, const int channels, float* chan_means);

__global__ void varianceS(float* in, const int N, const int channels, const float* chan_means, float* chan_vars);


__device__ double warpReduceSumD(double val);

__device__ double blockReduceSumD(double val);

__global__ void multiplyD(double* input, double* output, double alpha, int N);

__global__ void meanD(double* in, const int N, const int channels, double* chan_means);

__global__ void varianceD(double* in, const int N, const int channels, const double* chan_means, double* chan_vars);