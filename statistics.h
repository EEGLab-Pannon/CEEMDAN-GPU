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

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

/* warp-level reduction functions */
__inline__ __device__
float warpReduceSum(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  return val;
}

__inline__ __device__
float4 warpReduceSum(float4 val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    val.x += __shfl_xor_sync(0xffffffff, val.x, offset, 32);
    val.y += __shfl_xor_sync(0xffffffff, val.y, offset, 32);
    val.z += __shfl_xor_sync(0xffffffff, val.z, offset, 32);
    val.w += __shfl_xor_sync(0xffffffff, val.w, offset, 32);
  }
  return val;
}

__inline__ __device__
float warpReduceMin(float val) {
  //for (int offset = warpSize/2; offset > 0; offset /= 2) 
  //  val = fmin(__shfl_xor_sync(0xffffffff, val, offset, 32), val);

	val = fmin(__shfl_xor_sync(0xffffffff, val, 16, 32), val);
	val = fmin(__shfl_xor_sync(0xffffffff, val, 8, 32), val);
	val = fmin(__shfl_xor_sync(0xffffffff, val, 4, 32), val);
	val = fmin(__shfl_xor_sync(0xffffffff, val, 2, 32), val);
	val = fmin(__shfl_xor_sync(0xffffffff, val, 1, 32), val);
	return val;
}

__inline__ __device__
float warpReduceMax(float val) {
 // for (int offset = warpSize/2; offset > 0; offset /= 2) 
	//val = fmax(__shfl_xor_sync(0xffffffff, val, offset, 32), val);
	val = fmax(__shfl_xor_sync(0xffffffff, val, 16, 32), val);
	val = fmax(__shfl_xor_sync(0xffffffff, val, 8, 32), val);
	val = fmax(__shfl_xor_sync(0xffffffff, val, 4, 32), val);
	val = fmax(__shfl_xor_sync(0xffffffff, val, 2, 32), val);
	val = fmax(__shfl_xor_sync(0xffffffff, val, 1, 32), val);
  return val;
}

__inline__ __device__
float2 warpReduceMinMax(float2 val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2){ 
	val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, offset, 32), val.x);
	val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, offset, 32), val.y);
  }
	//val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, 16, 32), val.x);
	//val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, 16, 32), val.y);
	//val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, 8, 32), val.x);
	//val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, 8, 32), val.y);
	//val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, 4, 32), val.x);
	//val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, 4, 32), val.y);
	//val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, 2, 32), val.x);
	//val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, 2, 32), val.y);
	//val.x = fmin(__shfl_xor_sync(0xffffffff, val.x, 1, 32), val.x);
	//val.y = fmax(__shfl_xor_sync(0xffffffff, val.y, 1, 32), val.y);

  return val;
}


/* block-level reduction functions */
__inline__ __device__
float blockReduceSum(float val) {
	/*static*/ __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

	return val;
}

__inline__ __device__
float4 blockReduceSum(float4 val) {
	static __shared__ float4 shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	if (threadIdx.x < blockDim.x / warpSize) {
		val = shared[lane];  
	}else{
		val.x = 0;
		val.y = 0;
		val.z = 0;
		val.w = 0;
	}

	if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

	return val;
}


__inline__ __device__
float blockReduceMin(float val) {
	/*static*/ __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMin(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceMin(val); //Final reduce within first warp

	return val;
}

__inline__ __device__
float blockReduceMax(float val) {
	static __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMax(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceMax(val); //Final reduce within first warp

	return val;
}

__inline__ __device__
float2 blockReduceMinMax(float2 val) {
	static __shared__ float2 shared[32]; // Shared mem for 32 partial values
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMinMax(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val.x = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].x : 0;
	val.y = (threadIdx.x < blockDim.x / warpSize) ? shared[lane].y : 0;

	if (wid==0) val = warpReduceMinMax(val); //Final reduce within first warp

	return val;
}

__global__
void multiply(float *input, float *output, float alpha, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid>=N)) return;
	//printf("tid: %d in: %f a: %f ", tid, input[tid], alpha);
	output[tid] = input[tid] * alpha;
	//printf("out: %f\n", output[tid]);
}

__global__
void multiply_skew(float *input, float *var, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid>=N)) return;
	//printf("tid: %d in: %f a: %f ", tid, input[tid], alpha);
	//	float factor = sqrtf((float)N*(N-1)) / (powf(variance,1.5) * N *(N-2));
	input[tid] *= sqrtf((float)N*(N-1)) / (powf(var[tid],1.5) * (N-2) * N);;
//	input[tid] /= powf(var[tid],1.5) * N;
	//printf("out: %f\n", output[tid]);
}

__global__
void multiply_kurtosis(float *input, float *var, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid>=N)) return;
	//printf("tid: %d in: %f a: %f ", tid, input[tid], alpha);
	input[tid] *= N*(N+1) / (var[tid]*var[tid] * (N-1) * (N-2) * (N-3));
	//printf("out: %f\n", output[tid]);
}
/////////////////////////////////////////////////////////////////////////////
/* global statistical kernels */

__global__ 
void mean(float *in, const int N, const int channels, float *chan_means)
{	
	extern __shared__ float temp[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	int count = 0;
	float sum = 0.0f;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;

	for (int i = tid; i < N; i+=grid_stride)
	{
		sum += in[ch_offset+i];
//		if (i==1) printf("ch:%d %d %.4f\n", ch, i, sum);
		count++;
	}
	// now compute the global sum for each channel by reduction
	temp[ch] = blockReduceSum(sum);
	
	__syncthreads();
	
	// store partial results divided 
	if (threadIdx.x == 0){
//		printf("block x: %d, block y: %d, threadidx: %d, block sum: %.f\n", blockIdx.x, blockIdx.y, threadIdx.x, temp[ch]);
		atomicAdd((float*)&(chan_means[ch]), temp[ch]);
	}
//	__threadfence();
	// TODO: change to ticket counting as in threadFence sample!

	// the final block should do the division by N
	//if (blockIdx.x == 0 && tid == 0){
	//	//printf("ch: %d N: %d mean: %.f\n", ch, N, chan_means[ch]/(float)N);
	//	chan_means[ch] /= N;
	//}
}

__global__ 
void variance(float *in, const int N, const int channels, const float *chan_means, float *chan_vars){
	extern __shared__ float temp[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	temp[ch] = 0.0f;

	float variance = 0.0f;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	float mean = chan_means[ch];
	for (int i = tid; i < N; i+=grid_stride)
	{
		variance += (in[ch_offset+i]-mean)*(in[ch_offset+i]-mean);
	}
	//variance += (in[tid]-mean)*(in[tid]-mean);
	//if (tid==0)
	//printf("ch: %d, tid: %d, var: %.4f\n", ch, tid, variance);

	//d_ch_means[ch] = sum / count;
	temp[ch] = blockReduceSum(variance);
		
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd((float*)&(chan_vars[ch]), temp[ch]); 

	//__threadfence();

	//if (blockIdx.x == 0 && tid == 0){
	//	//printf("ch: %d: reduced sum: %.4f\n", ch, temp[ch]);
	//	chan_vars[ch] /= (N-1);  
	//}
}

__global__ 
void avgDeviation(float *in, const int N, const int channels, const float *chan_means, float *chan_devs){
	extern __shared__ float temp[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	float dev = 0.0f;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	float mean = chan_means[ch];
	for (int i = tid; i < N; i+=grid_stride)
	{
		dev += fabs(in[ch_offset+i]-mean);
	}

	temp[ch] = blockReduceSum(dev);

	if (tid == 0)
		atomicAdd((float*)&(chan_devs[ch]), temp[ch]/N); 
}

__global__ 
void minimum(float *in, const int N, const int channels, float *chan_mins){
	extern __shared__ float temp[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	float temp_min = FLT_MAX;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	for (int i = tid; i < N; i+=grid_stride)
	{
		temp_min = fmin(in[ch_offset+i], temp_min);
		
	}

	temp[ch] = blockReduceMin(temp_min);
	__syncthreads();
	//if (temp[ch] == 0) printf("MIN ZERO\n");
	if (threadIdx.x == 0) chan_mins[ch] = temp[ch]; 
}

__global__ 
void maximum(float *in, const int N, const int channels, float *chan_maxs){
	extern __shared__ float temp[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	float temp_max = -FLT_MAX;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	for (int i = tid; i < N; i+=grid_stride)
	{
		temp_max = fmax(in[ch_offset+i], temp_max);
	}

	temp[ch] = blockReduceMax(temp_max);
	__syncthreads();

	if (threadIdx.x == 0) chan_maxs[ch] = temp[ch]; 
}

__global__ 
void minmax(float *in, const int N, const int channels, float2 *results){
	extern __shared__ float2 sh_minmax[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	float temp_min = FLT_MAX;
	float temp_max = -FLT_MAX;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	for (int i = tid; i < N; i+=grid_stride)
	{
		float value = in[ch_offset+i];
		temp_min = fmin(value, temp_min);
		temp_max = fmax(value, temp_max);
	}

	sh_minmax[ch].x = blockReduceMin(temp_min);
	sh_minmax[ch].y = blockReduceMax(temp_max);

	if (tid == 0) results[ch] = sh_minmax[ch]; 
}

__global__ 
void minmax2(float *in, const int N, const int channels, float2 *results){
	extern __shared__ float2 sh_minmax[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	float2 temp_minmax;
	temp_minmax.x = FLT_MAX;
	temp_minmax.y = -FLT_MAX;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	for (int i = tid; i < N; i+=grid_stride)
	{
		float value = in[ch_offset+i];
		temp_minmax.x = fmin(value, temp_minmax.x);
		temp_minmax.y = fmax(value, temp_minmax.y);
	}

	sh_minmax[ch] = blockReduceMinMax(temp_minmax);

	if (tid == 0) results[ch] = sh_minmax[ch]; 
}


__global__ 
void skew(float *in, const int N, const int channels, const float *chan_means, const float *chan_vars, float *chan_skewness){
	extern __shared__ float temp[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;

	float skew = 0.0;
	float mean = chan_means[ch];
	float variance = chan_vars[ch];
 /*   float sigma = sqrtf(variance);
    float s3 = sigma * sigma * sigma;*/
	for (int i = tid; i < N; i+=grid_stride)
	{
		float x = (in[ch_offset+i]-mean);            
        skew += x*x*x;            
	}
	//float factor = sqrtf((float)N*(N-1)) / (powf(variance,1.5) * N *(N-2));

	temp[ch] = blockReduceSum(skew);
	__syncthreads();
	//printf("block sum: %.4f, count: %d\n", chan_vars[ch], tid);
	if (threadIdx.x == 0)
		atomicAdd((float*)&(chan_skewness[ch]), temp[ch]); 
}

__global__ 
void kurtosis(float *in, const int N, const int channels, const float *chan_means, const float *chan_vars, float *chan_kurtosis){
	extern __shared__ float temp[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;

	double kurtosis = 0.0;
	double mean = chan_means[ch];
	float variance = chan_vars[ch];
 /*   float sigma = sqrtf(variance);
    float s3 = sigma * sigma * sigma;*/
	for (int i = tid; i < N; i+=grid_stride)
	{
		double x = (in[ch_offset+i]-mean);            
        kurtosis += x*x*x*x;            
	}
	//float factor = sqrtf((float)N*(N-1)) / (powf(variance,1.5) * N *(N-2));

	temp[ch] = blockReduceSum((float)kurtosis);
	__syncthreads();
	//printf("block sum: %.4f, count: %d\n", chan_vars[ch], tid);
	if (threadIdx.x == 0)
		atomicAdd((float*)&(chan_kurtosis[ch]), temp[ch]); 
}

__global__ 
void skewness_kurtosis(float *in, const int N, const int channels, const float *chan_means, const float *chan_vars, float *chan_skewness, float *chan_kurtosis){
	extern __shared__ float2 temp2[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;

	float mean = chan_means[ch];
	float variance = chan_vars[ch];
	float s = 0.0f;
	float skew = 0.0f;
	float kurtosis = 0.0f;
 /*   float sigma = sqrtf(variance);
    float s3 = sigma * sigma * sigma;*/
	for (int i = tid; i < N; i+=grid_stride){
		s = (in[ch_offset+i]-mean);            
        skew += s*s*s;
		kurtosis += s*s*s*s;
	}
	float skew_factor = sqrtf((float)N*(N-1)) / (powf(variance,1.5) * N *(N-2));

	temp2[ch].x = blockReduceSum(skew);
	temp2[ch].y = blockReduceSum(kurtosis);

	//printf("block sum: %.4f, count: %d\n", chan_vars[ch], tid);
	if (tid == 0){
		atomicAdd((float*)&(chan_skewness[ch]), skew_factor * temp2[ch].x /*/ (N*variance*sqrt(variance))*/); 
		atomicAdd((float*)&(chan_kurtosis[ch]), temp2[ch].y / (N*variance*variance)); 
	}
}



/*
*/
__global__ 
void moments(float *in, const int N, const int channels, float *ch_mean, float *ch_variance, float *ch_skewness, float *ch_kurtosis){
	extern __shared__ float4 temp4[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	// first pass: compute mean
	float s = 0.0f;
	for (int i = tid; i < N; i+=grid_stride){
		s += in[ch_offset+i];
	}
	// now compute the global sum for each channel by reduction
	temp4[ch].x = blockReduceSum(s);

	// store partial results divided by the number of samples
	if (threadIdx.x == 0)
		atomicAdd((float*)&(ch_mean[ch]), temp4[ch].x/N);
	
	// wait for all threads to complete
	__syncthreads();

	// second pass: compute higher moments
	float mean = ch_mean[ch];
	float4 moments;
	for (int i = tid; i < N; i+=grid_stride)
	{
		s = in[ch_offset+i]-mean;
		moments.x += s;			//ep
		moments.y += s*s;		//variance 
		moments.z += s*s*s;		//skewness 
		moments.w += s*s*s*s;	//kurtosis 
	}
	//printf("tid: %d, ch: %d, sum: %.4f\n", threadIdx.x, ch, sum);

	temp4[ch] = blockReduceSum(moments);
	//printf("block sum: %.4f, count: %d\n", d_ch_means[ch], count);
	//if (tid == 0){
	//	atomicAdd((float*)&(ch_variance[ch]), (temp4[ch].y - temp4[ch].x*temp4[ch].x/N) / (N-1));
	//	atomicAdd((float*)&(ch_skewness[ch]), temp4[ch].z/(N*var*stdev));
	//	atomicAdd((float*)&(ch_kurtosis[ch]), temp4[ch]/(N-1));
	//}
}