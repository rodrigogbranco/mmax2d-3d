/*Author: Rodrigo Gonçalves de Branco
Date: 24/03/2015
*/

#include <iostream>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

using namespace std;

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
void prefixsumJAxis(vector< vector< vector<long> > >& v, long N, long k) {
	for(long i = 0; i < N; i++) {
		for(long j = 1; j < N; j++) {
			v[i][j][k] += v[i][j-1][k];
		}
	}
}

/*
Calculating the prefix sum of a slice of cube(matrix(i,k) on column j) -> Side representation 
Complexity: O(n^2)
*/
void prefixsumKAxis(vector< vector< vector<long> > >& v, long N, long j) {
	for(long i = 0; i < N; i++) {
		for(long k = 1; k < N; k++) {
			v[i][j][k] += v[i][j][k-1];
		}
	}
}

/*Kadane Algorithm
Complexity: O(n)*/
long maxSubArraySum(vector<long> v, long N)
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
    //cout<<"where: "<<where<<" max: "<<max_so_far<<endl;

    return max_so_far;
} 

/*Auxiliary function -> Print Cube
Complexity: O(n^3)*/
void print(vector< vector< vector<long> > > v, long N) {
	for(long k = 0; k < N; k++) {
		for(long i = 0; i < N; i++) {
			for(long j = 0; j < N; j++) {
				cout<<v[i][j][k]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
	}	
}

int main() {
	//size of cube
	long N;
	cin>>N;

	//cube representation: O(n^3) of space
	vector< vector< vector<long> > > cube(N, vector< vector<long> >(N, vector<long>(N)));

	//Reading the values
	for(long k = 0; k < N; k++) {
		for(long i = 0; i < N; i++) {
			for(long j = 0; j < N; j++) {
				cin>>cube[i][j][k];
			}
		}
	}

	//cout<<"original:"<<endl;
	//print(cube,N);

	timespec time1, time2;

	clock_gettime(CLOCK_MONOTONIC, &time1);

	//Calculating First Prefix Sum
	for(long k = 0; k < N; k++) {
		//for each face
		prefixsumJAxis(cube,N,k);
	}

	//cout<<"first ps:"<<endl;
	//print(cube,N);

	//Calculating Second Prefix Sum
	for(long k = 0; k < N; k++) {
		//for sides (perpendicular faces)
		prefixsumKAxis(cube,N,k);
	}

	//cout<<endl<<"second ps:"<<endl;
	//print(cube,N);

	long max = -1e9;
	for(long g = 0; g < N; g++) {
		for(long h = g; h < N; h++) {
			//For each Cgh on face, we have to calculate (n*(n+1)/2) sub-deeps for that colum combination
			//Lets call r and t the boundaries. So, we have to calculate |Cghrt| elements => O(n^4)
			for(long r = 0; r < N; r++) {
				for(long t = r; t < N; t++) {
					//Build the vector for Kadane
					vector<long> temp(N,0);

					//cout<<"g:"<<g<<" h:"<<h<<" r:"<<r<<" t:"<<t<<endl;
				
					for(long i = 0; i < N; i++) {
						//X = Cghrt(T) - cghrt(R-1) -cghrt(G-1)
						long tmp1 = cube[i][h][t];
						long tmp2 = r > 0 ? cube[i][h][r-1] : 0;
						long tmp3 = g > 0 ? cube[i][g-1][t] : 0;

						//Maybe repeated elements were subtracted. If that is true, we need correct it!
						long tmp4 = r > 0 && g > 0 ? cube[i][g-1][r-1] : 0 ;

						temp[i] = tmp1 - tmp2 - tmp3 + tmp4;
						//cout<<tmp1<<" - "<<tmp2<<" - "<<tmp3<<" + "<<tmp4<<" = "<<temp[i]<<endl;
					}
					//cout<<endl;

					long tmp = maxSubArraySum(temp,N);
					//cout<<"max: "<<tmp<<endl<<endl;;


					if(tmp > max) {
						max = tmp;
					}
				}
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &time2);

	//printf("%ld %.9f\n",max,timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));

	printf("%.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));

	//cout<<max<<endl;

	return 0;
}
