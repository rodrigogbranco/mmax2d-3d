#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <climits>
#include <sys/time.h>
#include <stdlib.h>

double timeSpecToSeconds(struct timespec* ts)
{
    return ((double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0)*1000;
}

using namespace std;

void preffixsum(vector< vector<int> >&  v, int m) {
	for(int i = 0; i < m; i++) {
		for(int j = 1; j < m; j++) {
			v[i][j] += v[i][j-1];
		}
	}
}



int sequence(vector< vector<int> > const & v, int c, int h)
{
        int max_so_far  = INT_MIN, max_ending_here = INT_MIN;
 
        // OPTIONAL: These variables can be added in to track the position of the subarray
        // size_t begin = 0;
        // size_t begin_temp = 0;
        // size_t end = 0;
 
        for(int i = 0; i < v.size(); i++)
        {
		int value = v[i][h] - (c == 0 ? 0 : v[i][c-1]);

                if(max_ending_here < 0)
                {
                        max_ending_here = value;
 
                        // begin_temp = i;
                }
                else
                {
                        max_ending_here += value;
                }
 
                if(max_ending_here >= max_so_far )
                {
                        max_so_far  = max_ending_here;
 
                        // begin = begin_temp;
                        // end = i;
                }
        }
        return max_so_far ;
}

int main(int argc, char** argv) {
	int nl;
	scanf("%d",&nl);

	int el = nl*nl;

	vector< vector<int> > v(nl,vector<int>(nl,0));

	srand(time(NULL));

	for(int i = 0; i < nl; i++)
	{
		for(int j = 0; j < nl; j++) {
			scanf("%d",&v[i][j]);
		} 
	}

	timespec time1, time2;

	clock_gettime(CLOCK_MONOTONIC, &time1);

	preffixsum(v,nl);

	int maxsum = INT_MIN;

	for(int i = 0; i < nl; i++) {
		for(int j = i; j < nl; j++) {
			int x = sequence(v,i,j);

			if(x > maxsum)
				maxsum = x;
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &time2);

	//printf("mmax2dseq: %d, tempo: %.9fms\n",maxsum,timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));
	printf("%.9f\n",timeSpecToSeconds(&time2) - timeSpecToSeconds(&time1));

	return 0;
}
