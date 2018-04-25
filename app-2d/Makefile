all:
	#libdevil-dev freeglut3-dev packages necessary
	g++ -g -o imagerender imagerender.cpp -lGL -lglut -lIL -Wall -pedantic
	nvcc -g -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -o mmax2d_triang mmax2d_triang.cu
	#libmagick++-dev graphicsmagick-libmagick-dev-compat libwebp-dev packages necessary
	g++ -g -o genmatrixfromimage genmatrixfromimage.cpp -lGL -lGLU -lglut -lIL `Magick++-config --cppflags --cxxflags --ldflags --libs`
 