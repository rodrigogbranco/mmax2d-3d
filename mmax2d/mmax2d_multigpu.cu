#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cuda_util.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

#define THREAD_NUM 128
#define BLOCKS_SCHED 2
#define SIZE_WARP 32

__device__ inline int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}


__device__ inline int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}

__global__ void computeCgh(int * vetor, int * result, int numlinhas, int dev_id, int dev_count) {
	
	int total_comp = numlinhas*(numlinhas+1)/2;

	int tid = ((total_comp/dev_count)*dev_id) + threadIdx.x  + blockIdx.x * blockDim.x;  //Identificação da thread;

	int c =  row_index(tid,numlinhas);
	int h = column_index(tid,numlinhas);

	int max_so_far  = INT_MIN, max_ending_here = INT_MIN;

	extern __shared__ int max_block[];

	if(threadIdx.x == 0)
		max_block[0] = INT_MIN;

	__syncthreads();


	if(tid < total_comp && h >= c) {
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
			atomicMax(&max_block[0],max_so_far);
	}

	__syncthreads();

	if(threadIdx.x == 0)
		atomicMax(&result[0],max_block[0]);
}

int main() {
	int el;
	scanf("%d",&el);
	el *= el;

	int * vetor = (int*)malloc(el*sizeof(int));
	int * keys = (int*)malloc(el*sizeof(int));

	int numlinhas  = (int)sqrt(el);

	int j = 0;
	for(int i = 1; i < el+1; i++)
	{
		keys[i-1] = j;
		scanf("%d",&vetor[i-1]);
		if(i % numlinhas == 0)
			j++;
	}

	int devCount;
	HANDLE_ERROR( cudaGetDeviceCount(&devCount));

	thrust::host_vector<int> max_device(devCount);

	int global_max = -1;	

	#pragma omp parallel num_threads(devCount) default(shared)
	{
		const int dev_id = omp_get_thread_num();

		HANDLE_ERROR( cudaSetDevice(dev_id) );
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, dev_id);

		int total_comp = numlinhas*(numlinhas+1)/2;

		unsigned tnumb = total_comp / THREAD_NUM > 0 ? THREAD_NUM : 32;
		unsigned bnumb = ((int)(total_comp / tnumb / devCount)) + 1;

		dim3 threadsPorBloco(tnumb);
		dim3 blocosPorGrid(bnumb);

		thrust::device_vector<int> d_vetor(vetor, vetor + el);
		thrust::device_vector<int> d_keys(keys, keys + el);
		thrust::device_vector<int> d_preffixsum(el);
		thrust::device_vector<int> d_result(1);

		float time;
		cudaEvent_t start,stop;
		HANDLE_ERROR( cudaEventCreate(&start) );
		HANDLE_ERROR( cudaEventCreate(&stop) );
		HANDLE_ERROR( cudaEventRecord(start, 0) );

		thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(),d_vetor.begin(),d_preffixsum.begin());

		CudaCheckError();

		computeCgh<<<blocosPorGrid,threadsPorBloco,sizeof(int)>>>(thrust::raw_pointer_cast(d_preffixsum.data()),thrust::raw_pointer_cast(d_result.data()),numlinhas, dev_id, devCount);

		HANDLE_ERROR( cudaThreadSynchronize() );
	
		max_device[dev_id] = d_result[0];

		#pragma omp barrier

		#pragma omp single 
		{
			for(int i = 0; i < devCount; i++) {
				if(global_max < max_device[i])
					global_max = max_device[i];
			}
		}

		HANDLE_ERROR( cudaEventRecord(stop, 0) );
		HANDLE_ERROR( cudaEventSynchronize(stop) );
		HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

		#pragma omp single
		{
			//printf("\nO resultado e: %d\n",global_max);
			//printf("O tempo foi de: %.9f ms para a mmax2d\n", time);			
			printf("mmax2d_multigpu: %d, tempo: %.9fms\n",global_max,time);
		}
		
	}
	

	return 0;
}
