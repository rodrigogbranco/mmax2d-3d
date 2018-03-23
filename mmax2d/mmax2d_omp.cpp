#include <omp.h>
#include <stdio.h>
#include<math.h>
#include<stdlib.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

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

void preffixsum(int*  v, int m) {
	int i, j;
	for(i = 0; i < m; i++) {
		for(j = 1; j < m; j++) {
			v[i*m + j] += v[i*m + j - 1];
		}
	}
}

int main (int argc, char *argv[]) {
 int rank, numelem,maxsum = INT_MIN;

 scanf("%d",&numelem);

 int n = numelem;
 numelem *= numelem;

  int * VetorDados = (int*)malloc(numelem*sizeof(int));

  for(int i = 0; i < numelem; i++) {
  	 scanf("%d",&VetorDados[i]);	
  }

  timespec time1, time2;

  clock_gettime(CLOCK_MONOTONIC, &time1);
 
#pragma omp parallel private(rank) default(shared)
 {
  rank = omp_get_thread_num();
  int size = omp_get_num_threads();

  int totalcgh = n*(n+1)/2;

  //int particao = totalcgh % size != 0 ? (int)(totalcgh/size) + 1 : (int)(totalcgh/size);
  //particao = particao > 0 ? particao : 1;  
  
  //preffixsum
  #pragma omp for schedule(dynamic)
	for(int i = 0; i < n; i++) {
		for(int j = 1; j < n; j++) {
			VetorDados[i*n + j] += VetorDados[i*n + j - 1];
		}		
	}

   
   #pragma omp for schedule(dynamic)
   for(int p = 0; p < totalcgh; p++) {
	int tid = p;

	int c = row_index(tid,n);
	int h = column_index(tid,n);

	    int max_so_far  = INT_MIN;

	if(tid < (n*(n+1)/2) && h >= c) {
	    	int max_ending_here = INT_MIN;
        	int begin = 0;
       	 	int begin_temp = 0;
        	int end = 0;		

        	for(int i = 0; i < n; i++) {
			int value = VetorDados[i*n + h] - (c == 0 ? 0 : VetorDados[i*n + c - 1]);

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

		#pragma omp critical
		{
			if(max_so_far > maxsum) {
				maxsum = max_so_far;	
			}	
		}
	}
   }  
 }
 
clock_gettime(CLOCK_MONOTONIC, &time2);
 
 //printf("mmax2d_omp: %d Tempo gasto: %.9f ms\n",maxsum,timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));
 printf("%.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));

 free(VetorDados);
 
 return 0;
}
