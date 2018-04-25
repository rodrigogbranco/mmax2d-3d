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

struct Element{
  int qtdeEl;
  int c;
  int h;
  int begin;
  int end;
};

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

__global__ void computeCgh(int * vetor, int * result, Element* elementos, int numlinhas) {

	int tid = threadIdx.x  + blockIdx.x * blockDim.x;  //Identificação da thread;

	int c =  row_index(tid,numlinhas);
	int h = column_index(tid,numlinhas);

	int max_so_far  = INT_MIN, max_ending_here = INT_MIN;
        int begin = 0;
        int begin_temp = 0;
        int end = 0;

	if(tid < numlinhas*(numlinhas+1)/2 && h >= c) {
        	for(int i = 0; i < numlinhas; i++) {
			int value = vetor[i*numlinhas + h] - (c == 0 ? 0 : vetor[i*numlinhas + c - 1]);

                	if(max_ending_here < 0) {
                        	max_ending_here = value;
                         	begin_temp = i;
                	}
                	else {
                        	max_ending_here += value;
                	}
                	if(max_ending_here >= max_so_far ) {
                        	max_so_far  = max_ending_here;
                         	begin = begin_temp;
                      		end = i;
                	}
        	}
			result[tid] = max_so_far;
			elementos[tid].qtdeEl = (h - c + 1) * (end + 1 - begin);
			elementos[tid].c = c;
			elementos[tid].h = h;
			elementos[tid].begin = begin;
			elementos[tid].end = end;
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

	//printf("%d\n",el);

	int * vetor = (int*)malloc(el*sizeof(int));
	int * keys = (int*)malloc(el*sizeof(int));

	//printf("\nLendo os valores...\n");

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

	//printf("\nAlocando recursos...\n");

	thrust::device_vector<int> d_vetor(vetor, vetor + el);

	thrust::device_vector<int> d_keys(keys, keys + el);

	thrust::device_vector<int> d_preffixsum(el);

	thrust::device_vector<int> d_result(numlinhas*(numlinhas+1)/2);

	thrust::device_vector<Element> d_qtdeEl(numlinhas*(numlinhas+1)/2);


	float time,time2;
	cudaEvent_t start,stop, stop2;

	
    cudaDeviceProp devProp;

    cudaGetDeviceProperties(&devProp,0);

	unsigned tnumb = el / devProp.maxThreadsPerBlock > 0 ? devProp.maxThreadsPerBlock : 32;
	unsigned bnumb;

	if(el % tnumb == 0)
		bnumb = (int)(el / tnumb);
	else
		bnumb = ((int)(el / tnumb)) + 1;

	dim3 dimThreads(tnumb);
	dim3 dimBlocks(bnumb);


	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventCreate(&stop2) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );

	//printf("\nSoma de Prefixos...\n");

	thrust::inclusive_scan_by_key(d_keys.begin(), d_keys.end(),d_vetor.begin(),d_preffixsum.begin());

	//printf("Computando Cgh's...\n");

	computeCgh<<<dimBlocks,dimThreads>>>(thrust::raw_pointer_cast(d_preffixsum.data()),thrust::raw_pointer_cast(d_result.data()),thrust::raw_pointer_cast(d_qtdeEl.data()),numlinhas);

	HANDLE_ERROR( cudaThreadSynchronize() );

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

	thrust::sort_by_key(d_result.begin(), d_result.end(), d_qtdeEl.begin());

	HANDLE_ERROR( cudaThreadSynchronize() );

    HANDLE_ERROR( cudaEventRecord(stop2, 0) );
    HANDLE_ERROR( cudaEventSynchronize(stop2) );
    HANDLE_ERROR( cudaEventElapsedTime(&time2, start, stop2) );


	thrust::host_vector<int> h_result(d_result.begin(),d_result.end());
	thrust::host_vector<Element> h_el(d_qtdeEl.begin(),d_qtdeEl.end());

	int index = (numlinhas*(numlinhas+1)/2)-1;

	/*printf("\nO resultado e: %d com %d elementos\n",h_result[index],h_el[index].qtdeEl);
	printf("c=%d h=%d begin=%d end=%d\n",h_el[index].c,h_el[index].h,h_el[index].begin,h_el[index].end);
	printf("O tempo foi de: %.3f ms para a mmax2d\n", time);
	printf("O tempo foi de: %.3f ms para a mmax2d + thrust::sort\n", time2);*/

	printf("%d %d %d %d\n",h_el[index].c,h_el[index].h,h_el[index].begin,h_el[index].end);

	free(vetor);
	free(keys);
	//cudaFree(d_result);

	return 0;
}

