#include <iostream>
#include <cstdlib> 
#include <ctime>
using namespace std;

int main(int argc, char** argv) {
	long n;
	n = atol(argv[1]);

	cout<<n<<" ";

	srand(time(NULL));

	for(long i = 0; i < n*n*n; i++) {
		cout<<(rand() % 20 - 10)<<" ";
	}

	return 0;
}
