#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"
export PATH=$CUDA_PATH/bin:$PATH

python setup.py build_ext --inplace
rm -rf build


CUDA_ARCH="-gencode arch=compute_52,code=sm_52 -arch=sm_52"


# compile roi_pooling
cd model/roi_pooling/src
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o roi_pooling.cu.o roi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py

# compile roi_align
cd ../../
cd model/roi_align/src
echo "Compiling roi align kernels by nvcc..."
nvcc -c -o roi_align_kernel.cu.o roi_align_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
