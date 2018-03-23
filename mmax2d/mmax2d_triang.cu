#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <thrust/sort.h>
#include <climits>
#include <thrust/extrema.h>
#include "cuda_util.h"

__device__ int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}


__device__ int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}

__global__ void computeCgh(int * vetor, int * result, int numlinhas) {

	int tid = threadIdx.x  + blockIdx.x * blockDim.x;  //Identificação da thread;

	int c =  row_index(tid,numlinhas);
	int h = column_index(tid,numlinhas);

	int max_so_far  = INT_MIN, max_ending_here = INT_MIN;

	if(tid < numlinhas*(numlinhas+1)/2 && h >= c) {
        	for(int i = 0; i < numlinhas; i++) {
			int value = vetor[i*numlinhas + h] - (c == 0 ? 0 : vetor[i*numlinhas + c - 1]);

                	if(max_ending_here < 0) {
                        	max_ending_here = value;
                	}
                	else {
                        	max_ending_here += value;
                	}
                	if(max_ending_here >= max_so_far ) {
                        	max_so_far  = max_ending_here;
                	}
        	}
			result[tid] = max_so_far;
	}
}

template <typename Vector>
void print(const Vector& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << " ";
  std::cout << "\n";
}


int main(int argc, char** argv)
{
	int el;
	scanf("%d",&el);
	el *= el;

	int * vetor = (int*)malloc(el*sizeof(int));
	int * keys = (int*)malloc(el*sizeof(int));

    int numlinhas  = (int)sqrt(el);

	srand(time(NULL));

	int j = 0;
	for(int i = 1; i < el+1; i++)
	{
		keys[i-1] = j;
		scanf("%d",&vetor[i-1]);
		if(i % numlinhas == 0)
			j++;
	}

	thrust::device_vector<int> d_vetor(vetor, vetor + el);

	thrust::device_vector<int> d_keys(keys, keys + el);

	thrust::device_vector<int> d_preffixsum(el);

	thrust::device_vector<int> d_result(numlinhas*(numlinhas+1)/2);


	float time;
	cudaEvent_t start,stop;

	
    cudaDeviceProp devProp;

    cudaGetDeviceProperties(&devProp,0);

	unsigned tnumb = el / devProp.maxThreadsPerBlock > 0 ? devProp.maxThreadsPerBlock : 32;
	unsigned bnumb = ((int)(el / tnumb)) + 1;

	dim3 dimThreads(tnumb);
	dim3 dimBlocks(bnumb);


	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );

	thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(),d_vetor.begin(),d_preffixsum.begin());
	
	CudaCheckError();

	computeCgh<<<dimBlocks,dimThreads>>>(thrust::raw_pointer_cast(d_preffixsum.data()),thrust::raw_pointer_cast(d_result.data()),numlinhas);

	HANDLE_ERROR( cudaThreadSynchronize() );


	thrust::device_vector<int>::iterator iter = thrust::max_element(d_result.begin(), d_result.end());

	HANDLE_ERROR( cudaThreadSynchronize() );

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );


	int result = *iter;

	//printf("\nO resultado e: %d\n",result);
	//printf("O tempo foi de: %.9f ms para a mmax2d\n", time);	
	printf("%.9f\n",time);
	//printf("mmax2d_triang: %d, tempo: %.9fms\n",result,time);


	free(vetor);
	free(keys);
	//cudaFree(d_result);

	return 0;
}

