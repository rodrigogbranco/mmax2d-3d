/*Author: Rodrigo Gonçalves de Branco
Date: 24/03/2015
*/

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cstdio>
#include <cmath>
#include <climits>
#include <stdio.h>
#include "cuda_util.h"
using namespace std;


__global__
void prefixsumJAxis(int* v, int N)
{
	int sqrN = N*N;
	for(int k = blockIdx.x; k < N; k += gridDim.x) {
		for(int i = threadIdx.x; i < N; i += blockDim.x) {
			for(int j = 1; j < N; j++) {
				v[sqrN*k + N*i + j] += v[sqrN*k + N*i + j-1];
			}
		}
	}
}

__global__
void prefixsumKAxis(int* v, int N)
{
	int sqrN = N*N;
	for(int j = blockIdx.x; j < N; j += gridDim.x) {
		for(int i = threadIdx.x; i < N; i += blockDim.x) {
			for(int k = 1; k < N; k++) {
				v[sqrN*k + N*i + j] += v[sqrN*(k-1) + N*i + j];
			}
		}
	}
}

__device__
int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}


__device__
int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}

__device__
int maxSubArraySum(int* v, int N, int g, int h, int r, int t) {
	int max_so_far = 0, max_ending_here = 0;


	int sqrN = N*N;

	   for(int i = 0; i < N; i++)
	   {
		int tmp1 = v[sqrN*t + N*i + h];
		int tmp2 = r > 0 ? v[sqrN*(r-1) + N*i + h] : 0;
		int tmp3 = g > 0 ? v[sqrN*t + N*i + (g-1)] : 0;

		//Maybe repeated elements were subtracted. If that is true, we need correct it!
		int tmp4 = r > 0 && g > 0 ? v[sqrN*(r-1) + N*i + (g-1)] : 0 ;

		int temp = tmp1 - tmp2 - tmp3 + tmp4;

		//printf("g:%d h:%d r:%d t:%d => %d - %d - %d + %d = %d\n",g,h,r,t,tmp1,tmp2,tmp3,tmp4,temp);

		max_ending_here = max_ending_here + temp;

	     if(max_ending_here < 0)
		max_ending_here = 0;

	     if(max_so_far < max_ending_here)
		max_so_far = max_ending_here;
	    }

	    return max_so_far;
}

__global__
void computeCghrt(int* v, int N, int * result)
{
	int computationSize = (N*(N+1))>>1;
	int maxsofar = INT_MIN;
	//to cover all R e T index
	//printf("blk:%d thd:%d gridDim:%d blockDim:%d\n",blockIdx.x,threadIdx.x,gridDim.x,blockDim.x);

	for(int blkstep = 0; blkstep < computationSize; blkstep += gridDim.x) {
		int r = row_index(blockIdx.x + blkstep,N);
		int t = column_index(blockIdx.x + blkstep,N);

		if(r >= 0 && t >= 0 && r < N && t < N && r <= t) {
			//to cover all G e H index
			for(int thdstep = 0; thdstep < computationSize; thdstep += blockDim.x) {
				int g = row_index(threadIdx.x + thdstep,N);
				int h = column_index(threadIdx.x + thdstep,N);

				if(g >= 0 && h >= 0 && g < N && h < N && g <= h) {
					int newmax = maxSubArraySum(v,N,g,h,r,t);
					maxsofar = newmax > maxsofar ? newmax : maxsofar;
				}	
			}
		}
	}
	
	atomicMax(result,maxsofar);
}

/*void print(int* v, int N) {
	for(int k = 0; k < N; k++) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				cout<<v[N*N*k + N*i + j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}	
}*/

int main() {
	//size of cube
	int N;
	cin>>N;

	//cube representation: O(n^3) of space
	int* cube = (int*)malloc(N*N*N*sizeof(int**));

	//Reading the values
	for(int k = 0; k < N; k++) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				cin>>cube[N*N*k + N*i + j];
			}
		}
	}

	//cout<<"original:"<<endl;

	//print(cube,N);

	int* dcube, *dresult;
	HANDLE_ERROR( cudaMalloc( (void**)&dcube,  N*N*N*sizeof(int))); 
	HANDLE_ERROR( cudaMemcpy( dcube, cube, N*N*N*sizeof(int),cudaMemcpyHostToDevice ) );

	int minValue = INT_MIN;
	HANDLE_ERROR( cudaMalloc( (void**)&dresult, sizeof(int))); 
	HANDLE_ERROR( cudaMemcpy( dresult, &minValue, sizeof(int),cudaMemcpyHostToDevice ) );

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);

	dim3 dimThreads(256);
	dim3 dimBlocks(32*numSMs);

	cudaEvent_t start,stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );

	HANDLE_ERROR( cudaEventRecord(start, 0) );

	prefixsumJAxis<<<dimBlocks,dimThreads>>>(dcube,N);

	HANDLE_ERROR( cudaThreadSynchronize() );

	//HANDLE_ERROR( cudaMemcpy( cube, dcube,N*N*N*sizeof(int),cudaMemcpyDeviceToHost));

	//cout<<"first ps:"<<endl;

	//print(cube,N);

	prefixsumKAxis<<<dimBlocks,dimThreads>>>(dcube,N);

	HANDLE_ERROR( cudaThreadSynchronize() );

	//cout<<endl<<"second ps:"<<endl;

	//HANDLE_ERROR( cudaMemcpy( cube, dcube,N*N*N*sizeof(int),cudaMemcpyDeviceToHost));

	//print(cube,N);

	//cout<<"computation size: "<<N*(N+1)/2<<endl;

	computeCghrt<<<dimBlocks,dimThreads>>>(dcube,N,dresult);

	HANDLE_ERROR( cudaThreadSynchronize() );

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(start) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );

	float time;	

	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

	int result;
	HANDLE_ERROR( cudaMemcpy( &result, dresult, sizeof(int),cudaMemcpyDeviceToHost));

	free(cube);
	cudaFree(dcube);
	cudaFree(dresult);

	//cout<<result<<endl;

	//printf("%i %.9f\n",result,time);
	printf("%.9f\n",time);

	return 0;
}
