/*Author: Rodrigo Gonçalves de Branco
Date: 24/03/2015
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <math.h>

#define ROOT 0

/*Cube representation: three axis(i,j,k)
n(k) matrices of nxn(i,j)

i -> row of matrices. Direction: down
j -> column of matrices. Direction: right
k -> deep of matrices. Direction: back
*/


/*
Calculating the prefix sum of a slice of cube(matrix(i,j) with deep k) -> Face representation
Complexity: O(n^2)
*/
void prefixsumJAxis(long * v, long N, long k) {
	long sqrN = N*N;
	long i, j;
	for(i = 0; i < N; i++) {
		for(j = 1; j < N; j++) {
			v[sqrN*k + N*i + j] += v[sqrN*k + N*i + j-1];
		}
	}
}

/*
Calculating the prefix sum of a slice of cube(matrix(i,k) on column j) -> Side representation 
Complexity: O(n^2)
*/
void prefixsumKAxis(long * v, long N, long j) {
	long sqrN = N*N;
	long i, k;
	for(i = 0; i < N; i++) {
		for(k = 1; k < N; k++) {
			v[sqrN*k + N*i + j] += v[sqrN*(k-1) + N*i + j];
		}
	}
}

/*Kadane Algorithm
Complexity: O(n)*/
long maxSubArraySum(long * v, long N)
{
	long max_so_far = 0, max_ending_here = 0;
	long i;

	//long where = 0;
	for(i = 0; i < N; i++)
	{
		max_ending_here = max_ending_here + v[i];
		if(max_ending_here < 0)
			max_ending_here = 0;
		if(max_so_far < max_ending_here) {
			max_so_far = max_ending_here;
			//where = i;
		}
	}
	//printf("where: %ld max: %ld\n",where,max_so_far);

	return max_so_far;
} 

/*Auxiliary function -> Print Cube
Complexity: O(n^3)*/
void print(long * v, long N) {
	long k, i, j;
	for(k = 0; k < N; k++) {
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				printf("%ld ",v[N*N*k + N*i + j]);
			}
			printf("\n");
		}
		printf("\n");
	}	
}

/*Auxiliary function -> Find row index
Complexity: O(1)*/
long row_index( unsigned long i, unsigned long M ){
    double m = M;
    double row = (-2*m - 1 + sqrt( (4*m*(m+1) - 8*(double)i - 7) )) / -2;
    if( row == (double)(long) row ) row -= 1;
    return (unsigned long) row;
}

/*Auxiliary function -> Find column index
Complexity: O(1)*/
long column_index( unsigned long i, unsigned long M ){
    unsigned long row = row_index( i, M);
    return  i - M * row + row*(row+1) / 2;
}

int main(int argc, char *argv[]) {
	
	int rank, size;
	long *cube, N, k, i, j, *temp, g, h, r, t, max, sqrN, tmp1, tmp2, tmp3, tmp4, tmp, computationSize, y, 
		startIndex, globalmax = LONG_MIN;
	double time1, time2;
	float compSize;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//size of cube
	if(rank == ROOT) {
		scanf("%ld",&N);
	}

	MPI_Bcast(&N, 1, MPI_LONG, ROOT, MPI_COMM_WORLD);

	//printf("rank=%d N=%ld\n",rank,N*N*N);

	cube =  (long*)malloc(N*N*N*sizeof(long));	

	if(rank == ROOT) {
		//cube representation: O(n^3) of space
		//Reading the values
		for(k = 0; k < N; k++) {
			for(i = 0; i < N; i++) {
				for(j = 0; j < N; j++) {
					scanf("%ld",&cube[N*N*k + N*i + j]);
				}
			}
		}		
	}	

	MPI_Bcast(cube, N*N*N, MPI_LONG, ROOT, MPI_COMM_WORLD);

	//printf("rank=%d original:\n",rank);
	//print(cube,N);

	//Build the vector for Kadane
	temp = (long*)malloc(N*sizeof(long));

	MPI_Barrier(MPI_COMM_WORLD);
	time1 = MPI_Wtime();

	//Calculating First Prefix Sum
	for(k = 0; k < N; k++) {
		//for each face
		prefixsumJAxis(cube,N,k);
	}

	//printf("first ps:\n");
	//print(cube,N);

	//Calculating Second Prefix Sum
	for(k = 0; k < N; k++) {
		//for sides (perpendicular faces)
		prefixsumKAxis(cube,N,k);
	}

	//printf("\nsecond ps:\n");
	//print(cube,N);

	max = LONG_MIN;
	sqrN = N*N;

	computationSize = ((N*(N+1))>>1)/size;
	compSize =  (float)((N*(N+1))>>1)/size;

	if((float)computationSize != compSize)
		computationSize++;

	for(y = 0; y < computationSize; y++) {
		startIndex = rank*computationSize + y;	

		g = row_index(startIndex,N);
		h = column_index(startIndex,N);	

		//printf("rank=%d startIndex=%ld g=%ld h=%ld\n",rank,startIndex,g,h);

		if(g >= 0 && h >= 0 && g < N && h < N && g <= h) {

			//for(g = 0; g < N; g++) {
			//	for(h = g; h < N; h++) {
					//For each Cgh on face, we have to calculate (n*(n+1)/2) sub-deeps for that colum combination
					//We'll call r and t the boundaries. So, we have to calculate |Cghrt| elements => O(n^4)
					for(r = 0; r < N; r++) {
						for(t = r; t < N; t++) {
							//printf("g:%ld h:%ld r:%ld t:%ld\n",g,h,r,t);
				
							for(i = 0; i < N; i++) {
								temp[i] = 0;
								//X = Cghrt(T) - cghrt(R-1) -cghrt(G-1)
								tmp1 = cube[sqrN*t + N*i + h];
								tmp2 = r > 0 ? cube[sqrN*(r-1) + N*i + h] : 0;
								tmp3 = g > 0 ? cube[sqrN*t + N*i + (g-1)] : 0;

								//Maybe repeated elements were subtracted. If that is true, we need correct it!
								tmp4 = r > 0 && g > 0 ? cube[sqrN*(r-1) + N*i + (g-1)] : 0 ;

								temp[i] = tmp1 - tmp2 - tmp3 + tmp4;
								//printf("%ld - %ld - %ld + %ld = %ld\n",tmp1,tmp2,tmp3,tmp4,temp[i]);
							}
							//printf("\n");

							tmp = maxSubArraySum(temp,N);
							//printf("max: %ld\n\n",tmp);


							if(tmp > max) {
								max = tmp;
							}
						}
					}
			//	}
			//}
		}
	}

   	MPI_Reduce(&max, &globalmax, 1, MPI_LONG, MPI_MAX, ROOT, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = MPI_Wtime();

	if(rank == ROOT) {
		printf("rank=%d max=%ld time=%.9f\n",rank,globalmax,(time2 - time1)*1000);
	}

	//printf("%.9f\n",(time2 - time1)*1000);

	//printf("%ld\n",max);

	free(cube);
	free(temp);

	MPI_Finalize();
	return 0;
}
