#include <stdio.h>
#include <omp.h>
#include "cuda_util.h"
#include <thrust/device_vector.h>

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

__global__ void computeCgh(int * vetor, int * result, int n, int rank, int size) {
	
	int total_comp = n*(n+1)/2;

	int tid = ((total_comp/size)*rank) + threadIdx.x  + blockIdx.x * blockDim.x;  //Identificação da thread;

	int g =  row_index(tid,n);
	int h = column_index(tid,n);

	int max_so_far  = INT_MIN, max_ending_here = INT_MIN;

	extern __shared__ int max_block[];

	if(threadIdx.x == 0)
		max_block[0] = INT_MIN;

	__syncthreads();


	if(tid < total_comp && h >= g) {
		//printf("rank=%d tid=%d Cgh=%d,%d\n",rank,tid,g,h);
        	for(int i = 0; i < n; i++) {
			int value = vetor[i*n + h] - (g == 0 ? 0 : vetor[i*n + g - 1]);

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
	int n;
	scanf("%d",&n);
	int sqrN = n*n;

	int * v;
	int * keys;

	//device 0 by default
	cudaMallocManaged(&v, sqrN*sizeof(int));
	cudaMallocManaged(&keys, sqrN*sizeof(int));

	int j = 0;
	for(int i = 1; i < sqrN+1; i++)
	{
		keys[i-1] = j;
		scanf("%d",&v[i-1]);
		if(i % n == 0)
			j++;
	}

	/*for(int i = 1; i <= sqrN; i++) {
		printf ("%d ",v[i-1]);

		if(i % n == 0)
			printf("\n");
	}

	printf("\n");*/

	int size;
	HANDLE_ERROR( cudaGetDeviceCount(&size));

	int * globalmax;
	cudaMallocManaged(&globalmax, size*sizeof(int));

	#pragma omp parallel num_threads(size) default(shared)
	{
		const int rank = omp_get_thread_num();

		int partition = n % size != 0 ? (int)(n/size) + 1 : n/size;
		partition = partition != 0 ? partition : 1; 

		HANDLE_ERROR( cudaSetDevice(rank) );

		int total_comp = n*(n+1)/2;

		unsigned tnumb = total_comp / THREAD_NUM > 0 ? THREAD_NUM : 32;
		unsigned bnumb = ((int)(total_comp / tnumb / size)) + 1;

		dim3 threadsPorBloco(tnumb);
		dim3 blocosPorGrid(bnumb);

		int startIndex = rank * partition * n;
		int endIndex = rank != size - 1 ? (rank+1) * partition * n : sqrN + 1;

		//printf("rank=%d partition=%d si=%d ei=%d\n",rank,partition,startIndex,endIndex);

		float time;
		cudaEvent_t start,stop;
		HANDLE_ERROR( cudaEventCreate(&start) );
		HANDLE_ERROR( cudaEventCreate(&stop) );
		HANDLE_ERROR( cudaEventRecord(start, 0) );

		thrust::inclusive_scan_by_key(keys + startIndex, keys + endIndex, v + startIndex ,v + startIndex);

		CudaCheckError();

		#pragma omp barrier

		computeCgh<<<blocosPorGrid,threadsPorBloco,sizeof(int)>>>(v,&globalmax[rank],n, rank, size);

		HANDLE_ERROR( cudaThreadSynchronize() );

		#pragma omp barrier

		for(int i = 1; i < size; i++) {
			if(globalmax[i] > globalmax[0]) {
				globalmax[0] = globalmax[i];				
			}
		}

		HANDLE_ERROR( cudaEventRecord(stop, 0) );
		HANDLE_ERROR( cudaEventSynchronize(stop) );
		HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

		#pragma omp single
		{
			//printf("\nO resultado e: %d\n",globalmax[0]);
			//printf("O tempo foi de: %.9f ms para a mmax2d\n", time);			
			printf("mmax2d_multigpu_um: %d, tempo: %.9fms\n",globalmax[0],time);
		}		
	}
	

	cudaSetDevice(0);



	/*for(int i = 1; i <= sqrN; i++) {
		printf ("%d ",v[i-1]);

		if(i % n == 0)
			printf("\n");
	}

	printf("\n");

	for(int i = 0; i < size; i++) {
		printf("%d\n",globalmax[i]);
	}*/

	cudaFree(v);
	cudaFree(keys);
	cudaFree(globalmax);
	
	return 0;
}
