#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <climits>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

using namespace std;

int row_index( unsigned int i, unsigned int M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(int) row ) row -= 1;
    return (unsigned int) row;
}

int column_index( unsigned int i, unsigned int M ){
    unsigned int row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}

void prefixsumJAxis(int* v, int N, int threadIdx, int threadDim)
{
	int sqrN = N*N;
	for(int k = threadIdx; k < N; k += threadDim) {
		for(int i = 0; i < N; i++) {
			for(int j = 1; j < N; j++) {
				v[sqrN*k + N*i + j] += v[sqrN*k + N*i + j-1];
			}
		}
	}
}

void prefixsumKAxis(int* v, int N, int threadIdx, int threadDim)
{
	int sqrN = N*N;
	for(int j = threadIdx; j < N; j += threadDim) {
		for(int i = 0; i < N; i++) {
			for(int k = 1; k < N; k++) {
				v[sqrN*k + N*i + j] += v[sqrN*(k-1) + N*i + j];
			}
		}
	}
}

void print(int* v, int N) {
	for(int k = 0; k < N; k++) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				cout<<v[N*N*k + N*i + j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}	
}

int maxSubArraySum(int* v, int N, int g, int h, int r, int t) {
	int max_so_far = 0, max_ending_here = 0;


	int sqrN = N*N;

	   for(int i = 0; i < N; i++)
	   {
		int tmp1 = v[sqrN*t + N*i + h];
		int tmp2 = r > 0 ? v[sqrN*(r-1) + N*i + h] : 0;
		int tmp3 = g > 0 ? v[sqrN*t + N*i + (g-1)] : 0;

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

int computeCghrt(int* v, int N, int threadIdx, int threadDim)
{
	int computationSize = (N*(N+1))>>1;
	int maxsofar = INT_MIN;

	for(int blkstep = 0; blkstep < computationSize; blkstep += threadDim) {
		int r = row_index(threadIdx + blkstep,N);
		int t = column_index(threadIdx + blkstep,N);

		if(r >= 0 && t >= 0 && r < N && t < N && r <= t) {
			//for(int thdstep = 0; thdstep < computationSize; thdstep += threadDim) {
			for(int g = 0; g < N; g++) {
				for(int h = g; h < N; h++) {
					int newmax = maxSubArraySum(v,N,g,h,r,t);
					maxsofar = newmax > maxsofar ? newmax : maxsofar;					
				}
			}
		}
	}
	
	return maxsofar;
}

int main() {
	int N;
	cin>>N;

	int* cube = (int*)malloc(N*N*N*sizeof(int**));

	for(int k = 0; k < N; k++) {
		for(int i = 0; i < N; i++) {
			for(int j = 0; j < N; j++) {
				cin>>cube[N*N*k + N*i + j];
			}
		}
	}

	int rank, size, globalmax = INT_MIN;

	timespec time1, time2;

	clock_gettime(CLOCK_MONOTONIC, &time1);

	#pragma omp parallel default(shared) private(rank, size)
 	{
		size = omp_get_num_threads();

  		rank = omp_get_thread_num();

		prefixsumJAxis(cube,N,rank,size);

		#pragma omp barrier

		prefixsumKAxis(cube,N,rank,size);

		#pragma omp barrier

		int localmax = computeCghrt(cube,N,rank,size);
                
                #pragma omp barrier

		#pragma omp critical
		{
			globalmax = localmax > globalmax ? localmax : globalmax;
			//cout<<"thread: "<<rank<<" size: "<<size<<" localmax: "<<localmax<<endl;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &time2);

	//print(cube,N);

	//printf("%i %.9f\n",globalmax,timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));
	printf("%.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));


	free(cube);

	return 0;
}
