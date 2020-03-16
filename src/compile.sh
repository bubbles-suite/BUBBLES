#!/bin/bash

nvcc -D_FORCE_INLINES -arch=sm_20 -use_fast_math -O2 -lcurand --ptxas-options -v mdgpu_np.cu -o mdgpu_np

cp mdgpu_np ..
