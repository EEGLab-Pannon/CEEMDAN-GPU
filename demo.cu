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

#include "cudaCEEMDAN.h"
#include "cudaCEEMDAN.cu"
#include "statistics.h"

int getBinSize(char* path);
void readBin(char* path, char* buf, size_t size);
void writeBin(char* path, char* buf, size_t size);

int main()
{
    //configuration for the input signal
    size_t numNoise = 20;
    size_t num_IMFs = 2;
    size_t max_iter = 300;
    float noiseStrength = 0.2;

    float* h_y, * d_y;
    int* h_x, * d_x;

    // get data size
    char filePathInput[] = "E:\\Software\\Nvidia_Development\\Project\\CUDA_CEEMDAN\\CEEMDAN\\ceemd_paper_example\\twoFrequencySignal.bin";
    size_t nbytes = getBinSize(filePathInput); // in bytes
    const size_t SignalLength = nbytes / sizeof(float);
    size_t oneChannelNbytes_coord = SignalLength * sizeof(int);
    size_t oneChannelNbytes_real = SignalLength * sizeof(float);

    // allocate array on host
    h_x = (int*)malloc(oneChannelNbytes_coord);
    h_y = (float*)malloc(oneChannelNbytes_real);

    // allocate array on device
    cudaMalloc((void**)&d_x, oneChannelNbytes_coord);
    cudaMalloc((void**)&d_y, oneChannelNbytes_real);

    // load data
    char* buf = (char*)malloc(oneChannelNbytes_real);
    readBin(filePathInput, buf, oneChannelNbytes_real);
    h_y = (float*)buf;

    // generate data index
    for (size_t i = 0; i < SignalLength; i++) {
        h_x[i] = (int)i;
    }

    // copy data to device
    cudaMemcpy(d_x, h_x, oneChannelNbytes_coord, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, oneChannelNbytes_real, cudaMemcpyHostToDevice);

    //ceemdan processing
    float* d_IMFs = NULL;
    cudaMalloc(&d_IMFs, num_IMFs * SignalLength * sizeof(float));
    float* IMFs = (float*)malloc(num_IMFs * SignalLength * sizeof(float));
    double exeTime = ceemdan(numNoise, SignalLength, num_IMFs, max_iter, d_x, d_y, d_IMFs, noiseStrength);
    printf("Execution time: %f \n", exeTime);

    //copy data back to host
    cudaMemcpy(IMFs, d_IMFs, num_IMFs * SignalLength * sizeof(float), cudaMemcpyDeviceToHost);
    char IMFs_file[] = "E:\\Software\\Nvidia_Development\\Project\\CUDA_CEEMDAN\\modes.bin";
    writeBin(IMFs_file, (char*)IMFs, num_IMFs * SignalLength * sizeof(float));
}

int getBinSize(char* path)
{
    int  size = 0;
    FILE* fp = fopen(path, "rb");
    if (fp)
    {
        fseek(fp, 0, SEEK_END);
        size = ftell(fp);
        fclose(fp);
    }
    //printf("\npath=%s,size=%d \n", path, size);
    return size;
}

void readBin(char* path, char* buf, size_t size)
{
    FILE* infile;
    if ((infile = fopen(path, "rb")) == NULL)
    {
        printf("\nCan not open the path: %s \n", path);
        exit(-1);
    }
    fread(buf, sizeof(char), size, infile);
    fclose(infile);
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
