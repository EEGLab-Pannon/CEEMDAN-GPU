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
//nvcc -arch=sm_70 -Xcompiler -fopenmp -lcublas -lcusparse -lcurand ./sample_synthetic_signal.cu ./cudaICEEMDAN.cu ./statistics.cu -o CUDA_ICEEMDAN

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cudaICEEMDAN.cuh"
#define PI 3.1415926

void writeBin(char* path, char* buf, size_t size);

int main()
{
	//configuration for the input signal
    size_t numNoise = 100;
    size_t num_IMFs = 2;
    size_t max_iter = 100;
    float noiseStrength = 0.2;
	const size_t SignalLength = 1000;
	
	//synthetic signal generation
	int* h_x = (int*)malloc(SignalLength * sizeof(int));
	float* h_y = (float*)malloc(SignalLength * sizeof(float));
	for (size_t i = 0; i < SignalLength; i++)
	{
		h_x[i] = (int)i;
        if ( (0 <= i && i < 500) || (748 < i && i < SignalLength))
            h_y[i] = sin(2 * PI * 0.065 * h_x[i]);
        else
            h_y[i] = sin(2 * PI * 0.065 * h_x[i]) + sin(2 * PI * 0.255 * (h_x[i] - 500));
    }

    // allocate array on device
    float* d_y; int* d_x;
    cudaMalloc((void**)&d_x, SignalLength * sizeof(int));
    cudaMalloc((void**)&d_y, SignalLength * sizeof(float));

    // copy data to device
    cudaMemcpy(d_x, h_x, SignalLength * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, SignalLength * sizeof(float), cudaMemcpyHostToDevice);

    //ceemdan processing
    float* d_IMFs = NULL;
    cudaMalloc(&d_IMFs, num_IMFs * SignalLength * sizeof(float));
    float* IMFs = (float*)malloc(num_IMFs * SignalLength * sizeof(float));
    double exeTime = iceemdanS(numNoise, SignalLength, num_IMFs, max_iter, d_x, d_y, d_IMFs, noiseStrength);
    printf("Execution time: %f \n", exeTime);

    //copy data back to host
    cudaMemcpy(IMFs, d_IMFs, num_IMFs * SignalLength * sizeof(float), cudaMemcpyDeviceToHost);
    char IMFs_file[] = "/path/to/modes.bin";
    writeBin(IMFs_file, (char*)IMFs, num_IMFs * SignalLength * sizeof(float));
}

void writeBin(char* path, char* buf, size_t size)
{
    FILE* outfile;
    if ((outfile = fopen(path, "wb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fwrite(buf, sizeof(char), size, outfile);
    fclose(outfile);
}
