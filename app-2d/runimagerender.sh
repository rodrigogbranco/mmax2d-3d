#!/bin/bash

./genmatrixfromimage $1 > imagemap.txt
./mmax2d_triang < imagemap.txt > numbers.txt
./imagerender $1 < numbers.txt


