#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
echo $TORCH
cd src

echo "Compiling correlation1d kernels by nvcc..."

rm correlation1d_cuda_kernel.o
rm -r ../_ext

nvcc -c -o correlation1d_cuda_kernel.o correlation1d_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
python build.py
