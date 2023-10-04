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

__device__ float warpReduceSumS(float val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_xor_sync(0xffffffff, val, offset, 32);
  return val;
}

__device__ float blockReduceSumS(float val) {
	/*static*/ __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSumS(val);     // Each warp performs partial reduction

	if (lane==0) shared[wid]=val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid==0) val = warpReduceSumS(val); //Final reduce within first warp

	return val;
}

__global__ void multiplyS(float* input, float* output, float alpha, int N){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid>=N)) return;
	//printf("tid: %d in: %f a: %f ", tid, input[tid], alpha);
	output[tid] = input[tid] * alpha;
	//printf("out: %f\n", output[tid]);
}

__global__ void meanS(float* in, const int N, const int channels, float* chan_means)
{	
	extern __shared__ float tempS[];
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
	tempS[ch] = blockReduceSumS(sum);
	
	__syncthreads();
	
	// store partial results divided 
	if (threadIdx.x == 0){
//		prcoord_tf("block x: %d, block y: %d, threadidx: %d, block sum: %.f\n", blockIdx.x, blockIdx.y, threadIdx.x, temp[ch]);
		atomicAdd((float*)&(chan_means[ch]), tempS[ch]);
	}
//	__threadfence();
	// TODO: change to ticket counting as in threadFence sample!

	// the final block should do the division by N
	//if (blockIdx.x == 0 && tid == 0){
	//	//prcoord_tf("ch: %d N: %d mean: %.f\n", ch, N, chan_means[ch]/(float)N);
	//	chan_means[ch] /= N;
	//}
}

__global__ void varianceS(float *in, const int N, const int channels, const float *chan_means, float *chan_vars){
	extern __shared__ float tempS[];
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid>=N) || (ch >= channels )) return;

	tempS[ch] = 0.0f;

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
	//prcoord_tf("ch: %d, tid: %d, var: %.4f\n", ch, tid, variance);

	//d_ch_means[ch] = sum / count;
	tempS[ch] = blockReduceSumS(variance);
		
	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd((float*)&(chan_vars[ch]), tempS[ch]);

	//__threadfence();

	//if (blockIdx.x == 0 && tid == 0){
	//	//prcoord_tf("ch: %d: reduced sum: %.4f\n", ch, temp[ch]);
	//	chan_vars[ch] /= (N-1);  
	//}
}


__device__ double warpReduceSumD(double val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val += __shfl_xor_sync(0xffffffff, val, offset, 32);
	return val;
}

__device__ double blockReduceSumD(double val) {
	/*static*/ __shared__ double shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSumD(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

	//read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) val = warpReduceSumD(val); //Final reduce within first warp

	return val;
}

__global__ void multiplyD(double* input, double* output, double alpha, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if ((tid >= N)) return;
	//printf("tid: %d in: %f a: %f ", tid, input[tid], alpha);
	output[tid] = input[tid] * alpha;
	//printf("out: %f\n", output[tid]);
}

__global__ void meanD(double* in, const int N, const int channels, double* chan_means)
{
	extern __shared__ double tempD[];
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid >= N) || (ch >= channels)) return;

	int count = 0;
	double sum = 0.0f;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;

	for (int i = tid; i < N; i += grid_stride)
	{
		sum += in[ch_offset + i];
		//		if (i==1) printf("ch:%d %d %.4f\n", ch, i, sum);
		count++;
	}
	// now compute the global sum for each channel by reduction
	tempD[ch] = blockReduceSumD(sum);

	__syncthreads();

	// store partial results divided 
	if (threadIdx.x == 0) {
		//		prcoord_tf("block x: %d, block y: %d, threadidx: %d, block sum: %.f\n", blockIdx.x, blockIdx.y, threadIdx.x, temp[ch]);
		atomicAdd((double*)&(chan_means[ch]), tempD[ch]);
	}
	//	__threadfence();
		// TODO: change to ticket counting as in threadFence sample!

		// the final block should do the division by N
		//if (blockIdx.x == 0 && tid == 0){
		//	//prcoord_tf("ch: %d N: %d mean: %.f\n", ch, N, chan_means[ch]/(float)N);
		//	chan_means[ch] /= N;
		//}
}

__global__ void varianceD(double* in, const int N, const int channels, const double* chan_means, double* chan_vars) {
	extern __shared__ double tempD[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ch = blockIdx.y;
	if ((tid >= N) || (ch >= channels)) return;

	tempD[ch] = 0.0f;

	double variance = 0.0f;
	int grid_stride = gridDim.x * blockDim.x;
	int ch_offset = ch * N;
	double mean = chan_means[ch];
	for (int i = tid; i < N; i += grid_stride)
	{
		variance += (in[ch_offset + i] - mean) * (in[ch_offset + i] - mean);
	}
	//variance += (in[tid]-mean)*(in[tid]-mean);
	//if (tid==0)
	//prcoord_tf("ch: %d, tid: %d, var: %.4f\n", ch, tid, variance);

	//d_ch_means[ch] = sum / count;
	tempD[ch] = blockReduceSumD(variance);

	__syncthreads();

	if (threadIdx.x == 0)
		atomicAdd((double*)&(chan_vars[ch]), tempD[ch]);

	//__threadfence();

	//if (blockIdx.x == 0 && tid == 0){
	//	//prcoord_tf("ch: %d: reduced sum: %.4f\n", ch, temp[ch]);
	//	chan_vars[ch] /= (N-1);  
	//}
}

